#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星影像分割测试脚本

基于SAM2 (Segment Anything Model 2) 对卫星影像进行自动分割，
获得原始的图像分割结果，为后续语义识别奠定基础。

功能特点：
1. 支持TIF格式的卫星影像输入
2. 自动生成分割掩码
3. 保存分割结果和可视化图像
4. 生成基础统计报告

作者: AI Assistant
创建日期: 2024
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 科学计算和图像处理库
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2

# 深度学习相关库
import torch

# SAM2相关模块
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# 配置文件
try:
    from config import MODEL
except ImportError:
    MODEL = {'large': 'path/to/your/model.pt'}  # 备用配置

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SatelliteImageProcessor:
    """
    卫星影像处理器类
    
    主要功能：
    1. 加载和预处理卫星影像
    2. 初始化SAM2模型
    3. 执行图像分割
    4. 生成基础统计信息
    """
    
    def __init__(self, model_type: str = "large", device: str = "auto"):
        """
        初始化卫星影像处理器
        
        Args:
            model_type: 模型类型 ('large', 'base', 'small', 'tiny')
            device: 计算设备 ('auto', 'cuda', 'cpu')
        """
        self.model_type = model_type
        self.model_config = self._get_config_path(model_type)
        self.device = self._get_device(device)
        self.sam2_model = None
        self.mask_generator = None
        
    def _get_config_path(self, model_type: str) -> str:
        """
        根据模型类型自动获取配置文件路径
        
        Args:
            model_type: 模型类型 ('large', 'base', 'small', 'tiny')
            
        Returns:
            str: 配置文件路径
        """
        # 获取当前脚本所在目录，作为项目根目录
        current_dir = Path(__file__).parent.absolute()
        
        # 模型类型到配置文件的映射（使用相对路径）
        model_config_map = {
            'large': 'sam2.1_hiera_l.yaml',
            'base': 'sam2.1_hiera_b+.yaml', 
            'small': 'sam2.1_hiera_s.yaml',
            'tiny': 'sam2.1_hiera_t.yaml'
        }
        
        config_filename = model_config_map.get(model_type)
        if not config_filename:
            logger.warning(f"未知的模型类型: {model_type}，使用默认large配置")
            config_filename = model_config_map['large']
        
        # 构建配置文件的完整路径
        config_path = current_dir / "sam2" / "configs" / "sam2.1" / config_filename
        
        # 检查配置文件是否存在
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            # 如果找不到，尝试寻找备用路径
            alternative_path = current_dir / "sam2" / "configs" / "sam2" / config_filename.replace("sam2.1_", "sam2_")
            if alternative_path.exists():
                logger.warning(f"使用备用配置文件: {alternative_path}")
                config_path = alternative_path
            else:
                raise FileNotFoundError(f"找不到配置文件: {config_path}")
            
        logger.info(f"使用配置文件: {config_path}")
        return str(config_path)
    
    def _get_device(self, device: str) -> torch.device:
        """
        获取计算设备
        
        Args:
            device: 设备类型字符串
            
        Returns:
            torch.device: PyTorch设备对象
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"使用CUDA设备: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = "mps" 
                logger.info("使用Apple MPS设备")
            else:
                device = "cpu"
                logger.info("使用CPU设备")
        
        device_obj = torch.device(device)
        
        # 设置CUDA优化选项
        if device_obj.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
        return device_obj
    
    def load_model(self, checkpoint_path: str):
        """
        加载SAM2模型
        
        Args:
            checkpoint_path: 模型权重文件路径
        """
        try:
            logger.info(f"正在加载SAM2模型: {checkpoint_path}")
            
            # 检查模型文件是否存在
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
            
            # 构建SAM2模型
            self.sam2_model = build_sam2(
                config_file=self.model_config,
                ckpt_path=checkpoint_path, 
                device=self.device,
                apply_postprocessing=False
            )
            
            # 创建自动掩码生成器
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2_model,
                points_per_side=32,        # 每边采样点数，影响分割精度
                points_per_batch=64,       # 批处理大小
                pred_iou_thresh=0.8,       # IoU阈值，过滤低质量掩码
                stability_score_thresh=0.9, # 稳定性分数阈值
                stability_score_offset=1.0,
                box_nms_thresh=0.7,        # NMS阈值，去除重复掩码
                crop_n_layers=1,           # 裁剪层数，提高小物体检测
                crop_nms_thresh=0.7,
                crop_overlap_ratio=0.34,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # 最小掩码区域面积
                use_m2m=True               # 启用掩码到掩码优化
            )
            
            logger.info("SAM2模型加载成功")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def load_satellite_image(self, image_path: str) -> np.ndarray:
        """
        加载卫星影像文件
        
        Args:
            image_path: 影像文件路径
            
        Returns:
            np.ndarray: RGB格式的影像数组 (H, W, 3)
        """
        try:
            logger.info(f"正在加载卫星影像: {image_path}")
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"影像文件不存在: {image_path}")
            
            # 获取文件扩展名
            file_ext = Path(image_path).suffix.lower()
            
            if file_ext == '.tif' or file_ext == '.tiff':
                # 处理TIF文件（可能是多波段）
                image = self._load_tif_image(image_path)
            else:
                # 处理其他格式图像
                image = Image.open(image_path)
                image = np.array(image.convert("RGB"))
            
            # 检查图像尺寸
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"图像格式错误，期望RGB格式 (H,W,3)，实际: {image.shape}")
            
            logger.info(f"影像加载成功，尺寸: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"影像加载失败: {str(e)}")
            raise
    
    def _load_tif_image(self, tif_path: str) -> np.ndarray:
        """
        加载TIF格式的卫星影像
        
        Args:
            tif_path: TIF文件路径
            
        Returns:
            np.ndarray: RGB格式的影像数组
        """
        try:
            # 尝试使用PIL加载
            image = Image.open(tif_path)
            
            # 如果图像有多个波段，选择前三个波段作为RGB
            if hasattr(image, 'n_frames') and image.n_frames > 1:
                # 多波段图像处理
                bands = []
                for i in range(min(3, image.n_frames)):
                    image.seek(i)
                    bands.append(np.array(image))
                
                # 组合RGB波段
                if len(bands) == 3:
                    rgb_image = np.stack(bands, axis=-1)
                elif len(bands) == 1:
                    # 单波段转为灰度图再转RGB
                    gray = bands[0]
                    rgb_image = np.stack([gray, gray, gray], axis=-1)
                else:
                    # 两个波段，添加一个零波段
                    zero_band = np.zeros_like(bands[0])
                    rgb_image = np.stack([bands[0], bands[1], zero_band], axis=-1)
            else:
                # 单波段或已经是RGB的图像
                rgb_image = np.array(image.convert("RGB"))
            
            # 数据类型转换和归一化
            if rgb_image.dtype == np.uint16:
                # 16位图像归一化到8位
                rgb_image = (rgb_image / 256).astype(np.uint8)
            elif rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
                # 浮点数图像归一化到0-255
                rgb_image = (rgb_image * 255).astype(np.uint8)
            
            return rgb_image
            
        except Exception as e:
            logger.error(f"TIF文件处理失败: {str(e)}")
            # 尝试使用opencv读取
            try:
                image = cv2.imread(tif_path, cv2.IMREAD_COLOR)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError("无法读取TIF文件")
            except:
                raise e
    
    def generate_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        生成图像分割掩码
        
        Args:
            image: RGB图像数组
            
        Returns:
            List[Dict]: 掩码列表，每个掩码包含分割信息
        """
        if self.mask_generator is None:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        logger.info("正在生成分割掩码...")
        masks = self.mask_generator.generate(image)
        logger.info(f"生成了 {len(masks)} 个分割掩码")
        
        return masks
    
    def process_masks(self, masks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理分割掩码，生成基础统计信息
        
        Args:
            masks: 分割掩码列表
            
        Returns:
            Dict: 处理结果，包含掩码信息和统计数据
        """
        logger.info("正在处理分割结果...")
        
        # 计算基础统计信息
        total_area = sum(mask['area'] for mask in masks)
        total_masks = len(masks)
        
        # 按面积排序
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        result = {
            'masks': sorted_masks,
            'statistics': {
                'total_masks': total_masks,
                'total_area': total_area,
                'largest_mask_area': sorted_masks[0]['area'] if sorted_masks else 0,
                'smallest_mask_area': sorted_masks[-1]['area'] if sorted_masks else 0,
                'average_mask_area': total_area / total_masks if total_masks > 0 else 0
            }
        }
        
        logger.info(f"分割处理完成，共 {total_masks} 个区域")
        return result
    
    def save_results(self, image: np.ndarray, process_result: Dict[str, Any], 
                    output_dir: str, base_name: str):
        """
        保存分析结果
        
        Args:
            image: 原始图像
            process_result: 处理结果
            output_dir: 输出目录
            base_name: 基础文件名
        """
        logger.info("正在保存结果...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存原始图像
        original_path = os.path.join(output_dir, f"{base_name}_original.jpg")
        Image.fromarray(image).save(original_path, quality=95)
        
        # 2. 生成和保存分割可视化图像
        segmentation_path = os.path.join(output_dir, f"{base_name}_segmentation.jpg")
        self._save_segmentation_visualization(image, process_result['masks'], segmentation_path)
        
        # 3. 保存统计报告
        stats_path = os.path.join(output_dir, f"{base_name}_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(process_result['statistics'], f, ensure_ascii=False, indent=2)
        
        # 4. 生成详细报告
        report_path = os.path.join(output_dir, f"{base_name}_report.txt")
        self._save_detailed_report(process_result['statistics'], report_path)
        
        logger.info(f"结果已保存到: {output_dir}")
        
    def _save_segmentation_visualization(self, image: np.ndarray, masks: List[Dict], 
                                       output_path: str):
        """
        保存分割可视化图像
        """
        plt.figure(figsize=(15, 10))
        plt.imshow(image)
        
        # 显示所有分割掩码
        for mask_info in masks:
            mask = mask_info['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            h, w = mask.shape
            mask_image = mask.reshape(h, w, 1) * color_mask.reshape(1, 1, -1)
            plt.imshow(mask_image)
        
        plt.title("卫星影像分割结果", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_detailed_report(self, statistics: Dict, output_path: str):
        """
        保存详细的分析报告
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("卫星影像分割分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"总分割区域数量: {statistics['total_masks']}\n")
            f.write(f"总分析面积: {statistics['total_area']:,.0f} 像素\n")
            f.write(f"最大区域面积: {statistics['largest_mask_area']:,.0f} 像素\n")
            f.write(f"最小区域面积: {statistics['smallest_mask_area']:,.0f} 像素\n")
            f.write(f"平均区域面积: {statistics['average_mask_area']:,.0f} 像素\n\n")
            
            f.write("=" * 50 + "\n")
            f.write("注：面积单位为像素，具体实际面积需要根据影像分辨率计算\n")
            f.write("此报告显示SAM2模型对影像的基础分割结果统计信息\n")


def validate_config(config: Dict[str, str]) -> Dict[str, str]:
    """
    验证配置参数
    
    Args:
        config: 配置字典
        
    Returns:
        Dict: 验证后的配置
        
    Raises:
        ValueError: 配置参数无效时抛出
    """
    # 必需参数
    if 'input' not in config:
        raise ValueError("必须提供输入图像路径 (input)")
    
    # 默认值设置
    validated_config = {
        'input': config['input'],
        'output': config.get('output', './output'),
        'model': config.get('model', 'large'),
        'device': config.get('device', 'auto')
    }
    
    # 参数验证
    valid_models = ['large', 'base', 'small', 'tiny']
    if validated_config['model'] not in valid_models:
        raise ValueError(f"无效的模型类型: {validated_config['model']}，支持的类型: {valid_models}")
    
    valid_devices = ['auto', 'cuda', 'cpu', 'mps']
    if validated_config['device'] not in valid_devices:
        raise ValueError(f"无效的设备类型: {validated_config['device']}，支持的类型: {valid_devices}")
    
    return validated_config


def main(config: Optional[Dict[str, str]] = None):
    """
    主函数 - 卫星影像分割分析流程
    
    Args:
        config: 可选的配置字典，如果提供则直接使用，否则解析命令行参数
                格式: {
                    'input': '输入图像路径',
                    'output': '输出目录',
                    'model': '模型类型',
                    'device': '计算设备'
                }
    
    示例用法：
    1. 命令行方式：
       python satellite_segmentation_test.py --input your_image.tif --output ./output --model large --device auto
    
    2. 程序调用方式：
       test_config = {
           "input": "path/to/image.tif",
           "output": "path/to/output",
           "model": "large",
           "device": "cuda"
       }
       main(test_config)
    """
    # 根据参数来源获取配置
    if config is not None:
        # 使用传入的配置字典
        logger.info("使用手动传入的配置参数")
        try:
            args_dict = validate_config(config)
        except ValueError as e:
            logger.error(f"配置参数错误: {str(e)}")
            return
    else:
        # 解析命令行参数
        logger.info("解析命令行参数")
        parser = argparse.ArgumentParser(description='卫星影像分割测试')
        parser.add_argument('--input', '-i', required=True, 
                           help='输入卫星影像路径 (支持TIF格式)')
        parser.add_argument('--output', '-o', default='./output',
                           help='输出目录 (默认: ./output)')
        parser.add_argument('--model', '-m', default='large',
                           choices=['large', 'base', 'small', 'tiny'],
                           help='使用的模型类型 (默认: large)')
        parser.add_argument('--device', '-d', default='auto',
                           choices=['auto', 'cuda', 'cpu', 'mps'],
                           help='计算设备 (默认: auto)')
        
        args = parser.parse_args()
        args_dict = {
            'input': args.input,
            'output': args.output,
            'model': args.model,
            'device': args.device
        }
    
    # 打印当前使用的配置
    logger.info("当前配置参数:")
    for key, value in args_dict.items():
        logger.info(f"  {key}: {value}")
    
    try:
        # 1. 获取模型路径
        model_path = MODEL.get(args_dict['model'])
        if not model_path or not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            logger.error("请在config.py中正确配置模型路径")
            return
        
        # 2. 初始化处理器
        logger.info("初始化卫星影像处理器...")
        processor = SatelliteImageProcessor(
            model_type=args_dict['model'],
            device=args_dict['device']
        )
        
        # 3. 加载模型
        processor.load_model(model_path)
        
        # 4. 加载卫星影像
        image = processor.load_satellite_image(args_dict['input'])
        
        # 5. 生成分割掩码
        masks = processor.generate_masks(image)
        
        # 6. 处理分割结果
        process_result = processor.process_masks(masks)
        
        # 7. 保存结果
        input_name = Path(args_dict['input']).stem
        processor.save_results(image, process_result, args_dict['output'], input_name)
        
        # 8. 打印统计信息
        stats = process_result['statistics']
        print("\n" + "="*50)
        print("分割分析结果统计")
        print("="*50)
        print(f"总分割区域: {stats['total_masks']}")
        print(f"总分析面积: {stats['total_area']:,.0f} 像素")
        print(f"最大区域面积: {stats['largest_mask_area']:,.0f} 像素")
        print(f"最小区域面积: {stats['smallest_mask_area']:,.0f} 像素")
        print(f"平均区域面积: {stats['average_mask_area']:,.0f} 像素")
        
        print("\n结果已保存到:", args_dict['output'])
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        if config is None:  # 只有命令行模式才退出程序
            sys.exit(1)


def run_test_with_config():
    """
    使用预设配置运行测试
    这是一个便于在开发环境中快速测试的函数
    """
    test_config = {
        "input": r"E:\CV\data\image\nanyuan_part.tif",
        "output": r"E:\CV\data\test_output",
        "model": "large", 
        "device": "cuda"
    }
    
    print("使用预设配置运行卫星影像分割测试")
    print("="*50)
    print("配置参数:")
    for key, value in test_config.items():
        print(f"  {key}: {value}")
    print("="*50)
    
    main(test_config)


if __name__ == "__main__":
    """
    程序入口点
    
    支持两种运行模式：
    1. 命令行模式 (无参数或带参数)：
       python satellite_segmentation_test.py --input your_image.tif --output ./output --model large --device cuda
    
    2. 测试配置模式 (修改下方test_config字典)：
       直接运行脚本使用预设配置
    """
    
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        # 有命令行参数，使用命令行模式
        main()
    else:
        # 无命令行参数，使用测试配置模式
        run_test_with_config()


