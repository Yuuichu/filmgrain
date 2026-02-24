# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**Film Grain Generator v2.0** - 真实胶片颗粒模拟工具

基于 [Dehancer](https://www.dehancer.com/learn/article/grain) 算法研究和 [City Frame Photography](https://cityframe-photography.com/blog/building-an-authentic-film-grain-simulator.html) 技术文档实现。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 GUI
python filmgrain_gui.py

# 命令行使用
python filmgrain.py input.jpg                    # 默认 ISO 400 彩色
python filmgrain.py input.jpg --iso 1600         # 高感光度
python filmgrain.py input.jpg --type bw          # 黑白胶片
python filmgrain.py input.jpg --type positive    # 正片模式
```

## v2.0 核心算法

### 1. 双峰尺寸分布
```python
# 70% 小颗粒 + 30% 大颗粒 (模拟真实银盐晶体混合)
grain = small_grain * 0.7 + large_grain * 0.3
```

### 2. Perlin 噪声聚簇
```python
# 颗粒形成群落，非均匀分布
cluster_map = perlin.fractal_noise(scale=30+iso/50, octaves=3)
grain *= (1 - cluster_strength) + cluster_strength * cluster_map
```

### 3. 分区亮度响应
| 区域 | 负片 | 正片 |
|------|------|------|
| 暗部 (0-0.3) | 0.8x | 1.2x |
| 中间调 (0.3-0.7) | 1.0x | 1.0x |
| 高光 (0.7-1.0) | 1.2x | 0.6x |

### 4. RGB 空间相关性
```python
# 彩色胶片染料云: 70% 共享 + 30% 独立
correlation = 0.7
r_grain = correlation * base + (1-correlation) * r_independent
```

## ISO 预设参数

| ISO | 强度 | 小颗粒 | 大颗粒 | 聚簇 |
|-----|------|--------|--------|------|
| 50 | 0.015 | 0.8x | 1.5x | 0.2 |
| 400 | 0.050 | 1.5x | 3.0x | 0.4 |
| 3200 | 0.140 | 3.0x | 6.0x | 0.7 |

## 胶片类型

| 类型 | 说明 | 暗部响应 | 高光响应 |
|------|------|----------|----------|
| `color` | 彩色负片 (Kodak Portra) | 1.0 | 0.8 |
| `bw` | 黑白负片 (Tri-X) | 1.1 | 0.7 |
| `negative` | 通用负片 | 0.8 | 1.2 |
| `positive` | 正片/幻灯片 | 1.2 | 0.6 |

## 部署

```bash
# 推送到 GitHub 后 Vercel 自动部署
git add -A && git commit -m "message" && git push
```

API: `https://filmgrain.vercel.app/api/grain`

## 项目结构

```
filmgrain/
├── filmgrain.py        # 核心算法 v2.0 + CLI
├── filmgrain_gui.py    # 桌面 GUI
├── api/
│   ├── grain.py        # Vercel API v2.0
│   └── index.html      # 在线测试页面
├── ios-shortcut/
│   └── README.md       # iOS 快捷指令指南
└── vercel.json
```
