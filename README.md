# Ultrasound Bubble Processing

这是一个用于超声波微泡信号处理的Python项目。该项目包含三个主要模块：滤波（filter）、定位（location）和跟踪（tracking），用于处理和分析超声波图像中的微泡信号。

## 项目结构

- `filter/` - 滤波模块
  - `fil_1.py` - 低通和高通Butterworth滤波器示例
  - `fil_2.py` - 其他滤波方法
  - `fil_3.py` - 高级滤波技术

- `loc/` - 定位模块
  - `loc_1.py` - 基于局部最大值的微泡定位
  - `loc_2.py` - 其他定位算法
  - `loc_3.py` - 定位优化
  - `loc_4.py` - 多尺度定位
  - `loc_5.py` - 定位验证
  - `loc_6.py` - 定位可视化

- `track/` - 跟踪模块
  - `tracking_1.py` - 基于匈牙利算法的微泡跟踪
  - `tra_2.py` - 跟踪算法改进
  - `tra_3.py` - 多目标跟踪

## 依赖项

- numpy
- matplotlib
- scipy
- scikit-image

## 安装

1. 确保您已安装Python 3.6或更高版本。
2. 安装依赖项：

```bash
pip install numpy matplotlib scipy scikit-image
```

## 使用方法

每个模块中的Python脚本都是独立的示例。您可以直接运行它们来查看结果：

```bash
python filter/fil_1.py
python loc/loc_1.py
python track/tracking_1.py
```

这些脚本将生成可视化图表，展示滤波、定位和跟踪的结果。

## 功能概述

### 滤波模块
- 实现各种数字滤波器来处理超声信号
- 支持低通、高通滤波等
- 用于去除噪声和提取有用信号

### 定位模块
- 在超声图像中检测微泡位置
- 使用图像处理技术如高斯滤波和峰值检测
- 提供精确的微泡定位算法

### 跟踪模块
- 跟踪微泡在连续帧中的运动
- 实现多目标跟踪算法
- 使用优化算法如匈牙利算法进行匹配

## 贡献

欢迎提交问题和拉取请求来改进这个项目。

## 许可证

本项目采用MIT许可证。