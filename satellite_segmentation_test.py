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
4. 生成纯色分区mask文件（PNG格式，与原影像严格套合）
5. 生成颜色映射表，记录分区ID与颜色的对应关系
6. 生成基础统计报告

输出文件说明：
- *_original.jpg: 原始输入影像
- *_segmentation.jpg: 分割可视化图像（原图+半透明彩色掩码）
- *_mask.png: 纯色分区mask文件（每个分区使用不同纯色）
- *_mask_color_mapping.json: 颜色映射表
- *_statistics.json: 统计数据
- *_report.txt: 详细分析报告

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
            
            # 创建自动掩码生成器，优化参数以降低内存使用
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2_model,
                points_per_side=16,        # 减少采样点数以降低内存使用
                points_per_batch=32,       # 减少批处理大小
                pred_iou_thresh=0.75,      # 略微降低IoU阈值
                stability_score_thresh=0.85, # 略微降低稳定性分数阈值
                stability_score_offset=1.0,
                box_nms_thresh=0.7,        # NMS阈值，去除重复掩码
                crop_n_layers=0,           # 禁用裁剪层以节省内存
                crop_nms_thresh=0.7,
                crop_overlap_ratio=0.34,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=200,  # 增加最小掩码区域面积，减少小碎片
                use_m2m=False              # 禁用掩码到掩码优化以节省内存
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
            
            logger.info(f"影像加载成功，原始尺寸: {image.shape}")
            
            # 检查图像是否过大，SAM2处理大图像时容易出现内存问题
            h, w = image.shape[:2]
            max_dimension = 1024  # SAM2推荐的最大尺寸，可根据内存情况调整
            
            if max(h, w) > max_dimension:
                # 计算缩放比例
                scale = max_dimension / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                logger.info(f"图像过大，将从 {w}x{h} 缩放到 {new_w}x{new_h} 以适配SAM2模型")
                
                # 使用高质量的缩放算法
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.info(f"图像缩放完成，新尺寸: {image.shape}")
            
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
        生成图像分割掩码，包含内存优化和错误恢复
        
        Args:
            image: RGB图像数组
            
        Returns:
            List[Dict]: 掩码列表，每个掩码包含分割信息
        """
        if self.mask_generator is None:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        logger.info("正在生成分割掩码...")
        
        # 尝试生成掩码，如果内存不足则进一步降低分辨率
        original_image = image.copy()
        current_image = image
        
        for attempt in range(3):  # 最多尝试3次
            try:
                h, w = current_image.shape[:2]
                logger.info(f"尝试 {attempt + 1}: 使用图像尺寸 {w}x{h}")
                
                masks = self.mask_generator.generate(current_image)
                logger.info(f"生成了 {len(masks)} 个分割掩码")
                
                # 如果当前图像被缩放了，需要将掩码坐标转换回原始尺寸
                if current_image.shape[:2] != original_image.shape[:2]:
                    masks = self._rescale_masks(masks, current_image.shape[:2], original_image.shape[:2])
                    logger.info("已将掩码坐标转换回原始图像尺寸")
                
                return masks
                
            except Exception as e:
                if "Unable to allocate" in str(e) and attempt < 2:
                    # 内存不足，进一步缩小图像
                    scale_factor = 0.7  # 每次缩小到原来的70%
                    h, w = current_image.shape[:2]
                    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                    current_image = cv2.resize(current_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    logger.warning(f"内存不足，进一步缩小图像到 {new_w}x{new_h}")
                    
                    # 强制垃圾回收
                    import gc
                    gc.collect()
                    continue
                else:
                    # 其他错误或已达到最大尝试次数
                    raise e
        
        # 如果所有尝试都失败了
        raise RuntimeError("无法生成掩码，即使在最低分辨率下也出现内存不足")
    
    def _rescale_masks(self, masks: List[Dict[str, Any]], 
                      current_shape: tuple, original_shape: tuple) -> List[Dict[str, Any]]:
        """
        将掩码从当前尺寸转换回原始图像尺寸
        
        Args:
            masks: 掩码列表
            current_shape: 当前图像尺寸 (h, w)
            original_shape: 原始图像尺寸 (h, w)
            
        Returns:
            List[Dict]: 转换后的掩码列表
        """
        current_h, current_w = current_shape
        original_h, original_w = original_shape
        
        scale_h = original_h / current_h
        scale_w = original_w / current_w
        
        rescaled_masks = []
        for mask in masks:
            # 复制掩码信息
            rescaled_mask = mask.copy()
            
            # 缩放分割掩码
            seg = mask['segmentation']
            rescaled_seg = cv2.resize(seg.astype(np.uint8), (original_w, original_h), 
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            rescaled_mask['segmentation'] = rescaled_seg
            
            # 缩放边界框
            if 'bbox' in mask:
                x, y, w, h = mask['bbox']
                rescaled_mask['bbox'] = [
                    x * scale_w, 
                    y * scale_h, 
                    w * scale_w, 
                    h * scale_h
                ]
            
            # 重新计算面积
            rescaled_mask['area'] = np.sum(rescaled_seg)
            
            rescaled_masks.append(rescaled_mask)
        
        return rescaled_masks
    
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
        
        # 3. 生成和保存纯色分区mask文件
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        self._save_colored_mask(image, process_result['masks'], mask_path)
        
        # 4. 保存统计报告
        stats_path = os.path.join(output_dir, f"{base_name}_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(process_result['statistics'], f, ensure_ascii=False, indent=2)
        
        # 5. 生成详细报告
        report_path = os.path.join(output_dir, f"{base_name}_report.txt")
        self._save_detailed_report(process_result['statistics'], report_path)
        
        logger.info(f"结果已保存到: {output_dir}")
        
    def _save_segmentation_visualization(self, image: np.ndarray, masks: List[Dict], 
                                       output_path: str):
        """
        保存分割可视化图像 - 优化内存使用
        """
        logger.info(f"正在生成分割可视化图像，共{len(masks)}个区域...")
        
        # 为了节省内存，如果图像过大，则降低分辨率
        h, w = image.shape[:2]
        max_dimension = 2048  # 最大尺寸限制
        
        if max(h, w) > max_dimension:
            # 计算缩放比例
            scale = max_dimension / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            logger.info(f"图像过大，将从 {w}x{h} 缩放到 {new_w}x{new_h} 以节省内存")
            
            # 缩放图像
            display_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            colored_mask = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            
            # 缩放掩码
            scaled_masks = []
            for mask_info in masks:
                original_mask = mask_info['segmentation']
                scaled_mask = cv2.resize(original_mask.astype(np.uint8), (new_w, new_h), 
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                scaled_masks.append({'segmentation': scaled_mask})
            masks_to_use = scaled_masks
        else:
            display_image = image
            colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
            masks_to_use = masks
        
        # 为每个区域分配颜色并合成到单一图像中
        for i, mask_info in enumerate(masks_to_use):
            if i % 50 == 0:  # 每处理50个掩码输出一次进度
                logger.info(f"处理进度: {i+1}/{len(masks_to_use)}")
            
            mask = mask_info['segmentation']
            # 生成随机颜色
            color = np.random.randint(0, 255, 3, dtype=np.uint8)
            
            # 将掩码区域着色
            colored_mask[mask] = color
        
        # 创建混合图像：原图像 + 半透明彩色掩码
        alpha = 0.5  # 透明度
        blended_image = (display_image.astype(np.float32) * (1 - alpha) + 
                        colored_mask.astype(np.float32) * alpha).astype(np.uint8)
        
        # 直接保存图像数组，不添加任何额外的标题或边框
        # 使用PIL保存，确保像素完全对应
        result_image = Image.fromarray(blended_image)
        result_image.save(output_path, quality=95)
        
        logger.info(f"分割结果图像保存完成: {output_path}")
        logger.info(f"输出图像尺寸: {blended_image.shape}")  # 输出尺寸信息用于验证
        
        # 强制释放内存
        del colored_mask, blended_image
        if 'scaled_masks' in locals():
            del scaled_masks
        if 'display_image' in locals() and display_image is not image:
            del display_image
        import gc
        gc.collect()
        
        logger.info("分割可视化图像生成完成")
    
    def _save_colored_mask(self, image: np.ndarray, masks: List[Dict], output_path: str):
        """
        生成和保存纯色分区mask文件
        
        每个分区使用不同的纯色表示，输出PNG格式文件
        背景区域为黑色(0,0,0)，每个分区使用唯一的颜色ID
        
        Args:
            image: 原始图像 (用于获取尺寸)
            masks: 分割掩码列表
            output_path: 输出PNG文件路径
        """
        logger.info(f"正在生成纯色分区mask文件，共{len(masks)}个区域...")
        
        h, w = image.shape[:2]
        
        # 创建空白的mask图像，初始化为黑色背景
        mask_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 为每个分区分配一个唯一的颜色ID
        # 使用HSV色彩空间生成均匀分布的颜色，确保视觉上容易区分
        num_masks = len(masks)
        
        # 生成颜色映射表，保存颜色ID到RGB的对应关系
        color_mapping = {}
        color_mapping[0] = (0, 0, 0)  # 背景色为黑色
        
        for i, mask_info in enumerate(masks):
            if i % 100 == 0:  # 每处理100个掩码输出一次进度
                logger.info(f"处理mask进度: {i+1}/{num_masks}")
            
            mask = mask_info['segmentation']
            
            # 为当前分区生成唯一颜色
            # 方法1: 使用HSV色彩空间生成均匀分布的色相
            if num_masks > 1:
                hue = int((i * 360) / num_masks)  # 色相值 0-359
                saturation = 255  # 饱和度固定为最大值
                value = 255      # 明度固定为最大值
                
                # 将HSV转换为RGB
                hsv_color = np.uint8([[[hue // 2, saturation, value]]])  # OpenCV的H范围是0-179
                rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
                color = tuple(rgb_color.astype(int))
            else:
                # 只有一个分区时，使用白色
                color = (255, 255, 255)
            
            # 确保颜色不与背景色重复
            if color == (0, 0, 0):
                color = (1, 1, 1)  # 如果生成了黑色，改为接近黑色的深灰色
            
            # 记录颜色映射关系
            color_id = i + 1  # 分区ID从1开始
            color_mapping[color_id] = color
            
            # 将掩码区域填充为对应颜色
            mask_image[mask] = color
        
        # 保存颜色映射文件，用于后续分析时对应分区
        mapping_path = output_path.replace('.png', '_color_mapping.json')
        mapping_data = {
            'total_regions': int(num_masks),  # 确保为Python原生int类型
            'background_color': [0, 0, 0],
            'region_colors': {
                str(region_id): {
                    'color_rgb': [int(c) for c in color],  # 确保颜色值为Python原生int类型
                    'region_id': int(region_id)  # 确保为Python原生int类型
                }
                for region_id, color in color_mapping.items() if region_id > 0
            },
            'description': '纯色分区mask文件的颜色映射表，每个颜色对应一个分割区域'
        }
        
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        
        # 保存mask图像为PNG格式（无损压缩，保持像素精确对应）
        result_image = Image.fromarray(mask_image, mode='RGB')
        result_image.save(output_path, format='PNG')
        
        logger.info(f"纯色分区mask文件保存完成: {output_path}")
        logger.info(f"颜色映射文件保存完成: {mapping_path}")
        logger.info(f"输出图像尺寸: {mask_image.shape}")
        logger.info(f"共生成 {num_masks} 个不同颜色的分区，背景为黑色")
        
        # 释放内存
        del mask_image
        import gc
        gc.collect()
    
    def _save_detailed_report(self, statistics: Dict, output_path: str):
        """
        保存详细的分析报告
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("卫星影像分割分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("分割统计信息:\n")
            f.write("-" * 30 + "\n")
            f.write(f"总分割区域数量: {statistics['total_masks']}\n")
            f.write(f"总分析面积: {statistics['total_area']:,.0f} 像素\n")
            f.write(f"最大区域面积: {statistics['largest_mask_area']:,.0f} 像素\n")
            f.write(f"最小区域面积: {statistics['smallest_mask_area']:,.0f} 像素\n")
            f.write(f"平均区域面积: {statistics['average_mask_area']:,.0f} 像素\n\n")
            
            f.write("输出文件说明:\n")
            f.write("-" * 30 + "\n")
            f.write("1. *_original.jpg: 原始输入影像\n")
            f.write("2. *_segmentation.jpg: 分割可视化图像（原图+半透明彩色掩码）\n")
            f.write("3. *_mask.png: 纯色分区mask文件（每个分区使用不同纯色）\n")
            f.write("4. *_mask_color_mapping.json: 颜色映射表（分区ID与颜色的对应关系）\n")
            f.write("5. *_statistics.json: 统计数据JSON文件\n")
            f.write("6. *_report.txt: 此详细报告文件\n\n")
            
            f.write("Mask文件使用说明:\n")
            f.write("-" * 30 + "\n")
            f.write("• mask.png文件中每个分区使用唯一的RGB颜色表示\n")
            f.write("• 背景区域为黑色(0,0,0)\n")
            f.write("• 分区颜色使用HSV色彩空间均匀分布生成，确保视觉区分度\n")
            f.write("• 颜色映射JSON文件记录了每个颜色对应的分区ID\n")
            f.write("• PNG格式无损保存，像素与原影像严格套合\n\n")
            
            f.write("=" * 50 + "\n")
            f.write("注：面积单位为像素，具体实际面积需要根据影像分辨率计算\n")
            f.write("此报告显示SAM2模型对影像的基础分割结果统计信息\n")
            f.write("mask文件可用于后续的语义识别和地物分类分析\n")


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
        logger.info(f"生成了 {len(masks)} 个分割掩码")
        
        # 6. 处理分割结果
        process_result = processor.process_masks(masks)
        
        # 释放原始掩码内存
        del masks
        import gc
        gc.collect()
        logger.info("已释放原始掩码内存")
        
        # 7. 保存结果
        input_name = Path(args_dict['input']).stem
        processor.save_results(image, process_result, args_dict['output'], input_name)
        
        # 8. 打印统计信息 (在释放内存前)
        stats = process_result['statistics']
        
        # 释放处理结果内存
        del process_result
        gc.collect()
        logger.info("已释放处理结果内存")
        print("\n" + "="*50)
        print("分割分析结果统计")
        print("="*50)
        print(f"总分割区域: {stats['total_masks']}")
        print(f"总分析面积: {stats['total_area']:,.0f} 像素")
        print(f"最大区域面积: {stats['largest_mask_area']:,.0f} 像素")
        print(f"最小区域面积: {stats['smallest_mask_area']:,.0f} 像素")
        print(f"平均区域面积: {stats['average_mask_area']:,.0f} 像素")
        
        print("\n输出文件列表:")
        print("-" * 30)
        print(f"1. 原始影像: {input_name}_original.jpg")
        print(f"2. 分割可视化: {input_name}_segmentation.jpg")
        print(f"3. 纯色分区mask: {input_name}_mask.png")
        print(f"4. 颜色映射表: {input_name}_mask_color_mapping.json")
        print(f"5. 统计数据: {input_name}_statistics.json")
        print(f"6. 详细报告: {input_name}_report.txt")
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
        "input": r"E:\CV\data\image\nanyuan_part1.tif",
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


