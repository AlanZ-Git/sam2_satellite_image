#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星影像分析系统测试脚本

用于测试系统基本功能和环境配置是否正确
"""

import os
import sys
import numpy as np
from PIL import Image
import torch

def test_environment():
    """测试Python环境和依赖库"""
    print("🔍 测试环境配置...")
    
    # 测试Python版本
    print(f"Python版本: {sys.version}")
    
    # 测试PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name()}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 测试必要的库
    required_libs = ['numpy', 'PIL', 'cv2', 'matplotlib']
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"✅ {lib}: 已安装")
        except ImportError:
            print(f"❌ {lib}: 未安装")
            return False
    
    return True

def test_config():
    """测试配置文件"""
    print("\n🔍 测试配置文件...")
    
    try:
        from config import MODEL
        print("✅ config.py: 加载成功")
        
        # 检查模型路径
        for model_name, model_path in MODEL.items():
            if os.path.exists(model_path):
                print(f"✅ {model_name}模型: {model_path}")
            else:
                print(f"❌ {model_name}模型: 路径不存在 {model_path}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ config.py: 导入失败 - {e}")
        return False

def test_sam2_import():
    """测试SAM2模块导入"""
    print("\n🔍 测试SAM2模块...")
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        print("✅ SAM2模块: 导入成功")
        return True
    except ImportError as e:
        print(f"❌ SAM2模块: 导入失败 - {e}")
        return False

def create_test_image():
    """创建测试用的模拟卫星影像"""
    print("\n🔧 创建测试影像...")
    
    # 创建一个模拟的卫星影像 (512x512)
    height, width = 512, 512
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 添加不同区域模拟不同地貌
    # 水体区域 (蓝色)
    image[50:150, 50:200] = [30, 100, 200]
    
    # 植被区域 (绿色)  
    image[200:350, 100:300] = [50, 150, 50]
    
    # 建筑区域 (灰色)
    image[300:450, 200:400] = [120, 120, 120]
    
    # 裸地区域 (棕色)
    image[100:200, 300:450] = [139, 100, 50]
    
    # 添加一些噪声使其更真实
    noise = np.random.randint(-20, 20, (height, width, 3))
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 保存测试图像
    test_image_path = "test_satellite_image.jpg"
    Image.fromarray(image).save(test_image_path)
    print(f"✅ 测试影像已创建: {test_image_path}")
    
    return test_image_path

def test_system_functionality():
    """测试系统主要功能"""
    print("\n🔍 测试系统功能...")
    
    try:
        # 导入主要类
        from satellite_segmentation_test import SatelliteImageProcessor
        
        # 初始化处理器（不加载模型，仅测试初始化）
        processor = SatelliteImageProcessor(device="cpu")  # 使用CPU避免GPU内存问题
        print("✅ 处理器初始化: 成功")
        
        # 测试图像加载功能
        test_image_path = create_test_image()
        image = processor.load_satellite_image(test_image_path)
        print(f"✅ 图像加载: 成功 - 尺寸 {image.shape}")
        
        # 清理测试文件
        os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ 系统功能测试: 失败 - {e}")
        return False

def run_quick_test():
    """运行快速功能测试（不需要模型）"""
    print("\n🚀 运行快速功能测试...")
    
    try:
        from satellite_segmentation_test import SatelliteImageProcessor
        
        # 创建测试图像
        test_image_path = create_test_image()
        
        # 初始化处理器
        processor = SatelliteImageProcessor(device="cpu")
        
        # 加载图像
        image = processor.load_satellite_image(test_image_path)
        
        # 测试基础处理功能（模拟掩码）
        # 创建一个简单的测试掩码
        test_masks = [{
            'segmentation': np.zeros((512, 512), dtype=bool),
            'area': 1000,
            'bbox': [50, 50, 100, 100]
        }]
        test_masks[0]['segmentation'][50:150, 50:150] = True
        
        # 测试掩码处理功能
        result = processor.process_masks(test_masks)
        
        print(f"✅ 掩码处理测试: 成功处理{result['statistics']['total_masks']}个区域")
        
        # 清理
        os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🛰️  卫星影像分割系统 - 环境测试")
    print("=" * 60)
    
    all_tests_passed = True
    
    # 1. 测试环境
    if not test_environment():
        all_tests_passed = False
    
    # 2. 测试配置
    if not test_config():
        all_tests_passed = False
    
    # 3. 测试SAM2导入
    if not test_sam2_import():
        all_tests_passed = False
    
    # 4. 测试系统功能
    if not test_system_functionality():
        all_tests_passed = False
    
    # 5. 运行快速测试
    if not run_quick_test():
        all_tests_passed = False
    
    # 总结
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 所有测试通过！系统配置正确。")
        print("\n📝 使用说明:")
        print("1. 准备一张TIF格式的卫星影像")
        print("2. 运行命令: python satellite_segmentation_test.py --input your_image.tif")
        print("3. 查看output目录中的分析结果")
    else:
        print("❌ 部分测试失败，请检查环境配置。")
        print("\n🔧 故障排除:")
        print("1. 确保安装了所有必要的Python库")
        print("2. 检查config.py中的模型路径")
        print("3. 确保SAM2环境正确安装")
    print("=" * 60)

if __name__ == "__main__":
    main()
