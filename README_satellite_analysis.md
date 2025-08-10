# 卫星影像分割测试系统

基于SAM2 (Segment Anything Model 2) 的卫星影像自动分割系统，用于获得原始的图像分割结果，为后续语义识别奠定基础。

## 功能特点

- 🛰️ **支持TIF格式**：专门优化处理卫星影像TIF文件，支持多波段数据
- 🎯 **智能分割**：基于SAM2的自动掩码生成，无需手动标注
- 📊 **统计分析**：提供详细的分割区域统计信息
- 🎨 **可视化输出**：生成分割结果可视化图像
- 📋 **报告生成**：自动生成JSON统计文件和文本报告

## 环境要求

- Python 3.8+
- PyTorch (支持CUDA加速)
- 已安装的SAM2项目依赖

## 安装说明

1. 确保SAM2项目环境已正确配置
2. 检查`config.py`中的模型路径设置：

```python
MODEL = {
    'large': r'E:\CV\model\large.pt'  # 修改为你的模型路径
}
```

## 使用方法

### 基本用法

```bash
python satellite_segmentation_test.py --input your_satellite_image.tif --output ./results
```

### 详细参数说明

```bash
python satellite_segmentation_test.py \
    --input /path/to/satellite_image.tif \    # 输入影像路径
    --output ./output \                        # 输出目录
    --model large \                           # 模型类型
    --device auto \                           # 计算设备 (auto/cuda/cpu)
    --config sam2/configs/sam2.1/sam2.1_hiera_l.yaml  # 模型配置
```

### 参数详解

- `--input, -i`: 输入卫星影像路径（必需）
  - 支持格式：TIF, TIFF, JPG, PNG等
  - 推荐使用TIF格式的卫星影像
  
- `--output, -o`: 输出目录（默认：./output）
  - 自动创建目录结构
  - 保存所有分析结果
  
- `--model, -m`: 模型类型（默认：large）
  - 目前支持：large
  - 对应config.py中的模型配置
  
- `--device, -d`: 计算设备（默认：auto）
  - auto: 自动选择最佳设备
  - cuda: 强制使用GPU
  - cpu: 使用CPU计算
  - mps: Apple Silicon Mac使用MPS
  
- `--config, -c`: SAM2模型配置文件路径
  - 默认使用large模型的配置文件

## 输出文件说明

运行完成后，会在输出目录生成以下文件：

```
output/
├── [filename]_original.jpg        # 原始影像
├── [filename]_segmentation.jpg    # 分割结果可视化
├── [filename]_statistics.json     # 统计数据(JSON格式)
└── [filename]_report.txt          # 详细分析报告
```

### 文件内容说明

1. **原始影像** (`_original.jpg`): 输入影像的JPG格式副本
2. **分割结果** (`_segmentation.jpg`): 显示所有分割区域的彩色掩码
3. **统计数据** (`_statistics.json`): 包含分割区域统计信息
4. **分析报告** (`_report.txt`): 人类可读的详细报告

## 分割功能说明

当前版本专注于基础分割功能：

- **自动分割**：SAM2自动识别图像中的不同区域
- **区域统计**：计算每个分割区域的面积和基本信息
- **可视化**：为每个分割区域分配随机颜色进行显示
- **统计报告**：生成分割区域数量、面积等基础统计信息

这为后续的语义识别和地貌分类奠定了基础。

## 性能优化建议

### GPU加速
- 推荐使用CUDA GPU加速处理
- 大型影像建议使用16GB+显存的GPU
- 支持自动混合精度计算

### 内存优化
- 大尺寸影像会消耗较多内存
- 建议系统内存16GB以上
- 可通过调整分割参数优化性能

### 参数调优

可在代码中调整以下参数来优化结果：

```python
# 在SatelliteImageProcessor类的load_model方法中
self.mask_generator = SAM2AutomaticMaskGenerator(
    model=self.sam2_model,
    points_per_side=32,        # 降低可提高速度，提高可增加精度
    points_per_batch=64,       # 根据显存大小调整
    pred_iou_thresh=0.8,       # 提高可减少低质量掩码
    stability_score_thresh=0.9, # 提高可减少不稳定掩码
    min_mask_region_area=100,  # 提高可过滤小区域
)
```

## 常见问题

### Q: 模型加载失败
A: 检查config.py中的模型路径是否正确，确保.pt文件存在

### Q: CUDA内存不足
A: 尝试降低points_per_side和points_per_batch参数，或使用CPU模式

### Q: TIF文件读取错误
A: 确保TIF文件格式正确，系统会自动处理多波段数据

### Q: 分类结果不准确
A: 可以调整landuse_color_ranges中的颜色范围参数

## 示例结果

典型的卫星影像分割结果：

```
==================================================
分割分析结果统计
==================================================
总分割区域: 156
总分析面积: 2,073,600 像素
最大区域面积: 45,230 像素
最小区域面积: 120 像素
平均区域面积: 13,292 像素
```

## 技术说明

### 算法流程

1. **影像预处理**: 加载TIF文件，处理多波段数据，转换为RGB格式
2. **自动分割**: 使用SAM2生成高质量分割掩码
3. **统计分析**: 计算每个分割区域的面积等基本信息
4. **结果输出**: 生成可视化图像和统计报告

### 分割算法

基于SAM2的自动掩码生成：
- 在图像上采样密集的点网格作为提示
- 为每个点生成高质量的分割掩码
- 使用NMS去除重复和低质量的掩码
- 支持多尺度分割以捕获不同大小的对象

## 扩展开发

系统采用模块化设计，便于扩展：

1. **添加语义分类**: 在`process_masks`方法基础上增加地貌分类逻辑
2. **改进分割算法**: 可调整SAM2的参数以优化不同类型影像的分割效果
3. **支持新格式**: 扩展`_load_tif_image`方法支持更多影像格式
4. **自定义可视化**: 修改`_save_segmentation_visualization`方法实现不同的显示效果

## 许可证

本项目基于SAM2开源协议，仅供学习和研究使用。
