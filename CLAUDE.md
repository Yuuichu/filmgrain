# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

胶片颗粒效果生成器 - 基于真实胶片特性的图像处理工具。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 GUI
python filmgrain_gui.py

# 命令行基础使用
python filmgrain.py input.jpg

# 指定 ISO 感光度
python filmgrain.py input.jpg --iso 1600

# 黑白胶片模式
python filmgrain.py input.jpg --bw

# 自定义输出路径和强度
python filmgrain.py input.jpg -o output.jpg --intensity 0.1
```

## GUI 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+O` | 打开图像 |
| `Ctrl+S` | 保存结果 |
| `空格` | 按住查看原图对比 |

## 核心算法

### 胶片颗粒模拟原理

1. **高斯噪声基底**: 使用正态分布生成基础颗粒
2. **尺寸缩放**: 根据 ISO 调整颗粒粒径 (低 ISO = 细颗粒)
3. **亮度响应**: 暗部颗粒更明显，使用 `1 - luminosity^0.6` 曲线
4. **色彩差异**:
   - 彩色: RGB 三通道独立噪声 (模拟染料云)
   - 黑白: 单通道噪声 + 硬边处理 (模拟银盐晶体)

### ISO 预设参数

| ISO  | 强度   | 颗粒尺寸 | 粗糙度 |
|------|--------|----------|--------|
| 50   | 0.02   | 1.0x     | 0.3    |
| 400  | 0.06   | 1.6x     | 0.6    |
| 3200 | 0.18   | 3.0x     | 0.9    |

## 部署到 Vercel (iOS 快捷指令)

```bash
# 推送到 GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/用户名/filmgrain.git
git push -u origin main

# 然后在 vercel.com 导入该仓库
```

部署后访问 `https://your-app.vercel.app` 查看在线测试页面。

API 端点: `POST /api/grain`

## 项目结构

```
filmgrain/
├── filmgrain.py        # 核心算法 + CLI
├── filmgrain_gui.py    # 桌面 GUI (tkinter)
├── api/
│   ├── grain.py        # Vercel serverless API
│   └── index.html      # 在线测试页面
├── ios-shortcut/
│   └── README.md       # iOS 快捷指令配置指南
├── vercel.json         # Vercel 部署配置
└── requirements.txt
```

## 扩展方向

- 添加 Perlin/Simplex 噪声选项
- 胶片品牌预设 (Kodak/Fuji 色调)
- 视频帧序列处理 (帧间颗粒变化)
- GPU 加速 (CUDA/OpenCL)
