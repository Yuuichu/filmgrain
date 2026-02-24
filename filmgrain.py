#!/usr/bin/env python3
"""
Film Grain Effect Generator v2.0
真实胶片颗粒模拟 - 基于专业研究数据

特性:
- 双峰尺寸分布 (70% 小颗粒 + 30% 大颗粒)
- Perlin 噪声聚簇 (有机随机分布)
- 分区亮度响应 (暗部/中间调/高光)
- RGB 通道空间相关性 (染料云模拟)
- 负片/正片模式
"""

import numpy as np
from PIL import Image, ImageFilter
import argparse
from pathlib import Path
from typing import Tuple, Optional
import math


class PerlinNoise:
    """Perlin 噪声生成器 - 真正的梯度噪声实现，用于有机聚簇效果"""

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        # 生成排列表
        p = np.arange(256, dtype=np.int32)
        self.rng.shuffle(p)
        self.perm = np.concatenate([p, p])  # 重复以避免溢出

        # 预计算梯度向量 (8个方向)
        self.gradients = np.array([
            [1, 1], [-1, 1], [1, -1], [-1, -1],
            [1, 0], [-1, 0], [0, 1], [0, -1]
        ], dtype=np.float32)

    def _fade(self, t):
        """平滑插值曲线 6t^5 - 15t^4 + 10t^3 (Perlin 改进版)"""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def noise2d(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """生成 2D Perlin 噪声"""
        # 网格单元坐标
        xi = np.floor(x).astype(np.int32)
        yi = np.floor(y).astype(np.int32)

        # 单元内相对坐标
        xf = x - xi
        yf = y - yi

        # 限制索引范围
        xi = xi & 255
        yi = yi & 255

        # 平滑插值权重
        u = self._fade(xf)
        v = self._fade(yf)

        # 四个角的哈希值 (逐元素计算)
        h00 = self.perm[self.perm[xi] + yi] & 7
        h01 = self.perm[self.perm[xi] + yi + 1] & 7
        h10 = self.perm[self.perm[xi + 1] + yi] & 7
        h11 = self.perm[self.perm[xi + 1] + yi + 1] & 7

        # 梯度点积
        g00 = self.gradients[h00, 0] * xf + self.gradients[h00, 1] * yf
        g10 = self.gradients[h10, 0] * (xf - 1) + self.gradients[h10, 1] * yf
        g01 = self.gradients[h01, 0] * xf + self.gradients[h01, 1] * (yf - 1)
        g11 = self.gradients[h11, 0] * (xf - 1) + self.gradients[h11, 1] * (yf - 1)

        # 双线性插值
        x1 = g00 + u * (g10 - g00)
        x2 = g01 + u * (g11 - g01)
        return x1 + v * (x2 - x1)

    def fractal_noise(self, width: int, height: int, scale: float = 50.0,
                      octaves: int = 4, persistence: float = 0.5) -> np.ndarray:
        """
        生成分形噪声 (多层 Perlin 叠加)

        Args:
            width, height: 输出尺寸
            scale: 基础缩放 (越大 = 聚簇越大)
            octaves: 叠加层数
            persistence: 高频衰减系数
        """
        # 创建坐标网格
        y_coords, x_coords = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing='ij'
        )

        noise = np.zeros((height, width), dtype=np.float32)
        amplitude = 1.0
        max_amplitude = 0.0
        freq = 1.0

        for _ in range(octaves):
            noise += amplitude * self.noise2d(
                x_coords * freq / scale,
                y_coords * freq / scale
            )
            max_amplitude += amplitude
            amplitude *= persistence
            freq *= 2

        return noise / max_amplitude


class FilmGrain:
    """
    真实胶片颗粒生成器 v2.1 - 分辨率自适应

    基于 Dehancer 算法研究和 City Frame Photography 技术文档
    颗粒尺寸会根据图像分辨率自动调整，确保视觉一致性
    """

    # 基准分辨率 (2K)
    REFERENCE_SIZE = 2048

    # ISO 预设: (intensity, small_grain_ratio, large_grain_ratio, cluster_strength)
    # grain_ratio 是相对于图像短边的比例
    ISO_PRESETS = {
        50:   (0.015, 0.0004, 0.0008, 0.2),   # 超细腻
        100:  (0.025, 0.0005, 0.0010, 0.25),  # 细腻
        200:  (0.035, 0.0006, 0.0013, 0.3),   # 细腻-中等
        400:  (0.050, 0.0008, 0.0016, 0.4),   # 标准
        800:  (0.070, 0.0010, 0.0020, 0.5),   # 明显
        1600: (0.100, 0.0013, 0.0026, 0.6),   # 粗颗粒
        3200: (0.140, 0.0016, 0.0032, 0.7),   # 超粗颗粒
    }

    # 胶片类型预设
    FILM_TYPES = {
        # (shadow_response, midtone_response, highlight_response, chroma_variance)
        "negative": (0.8, 1.0, 1.2, 0.15),   # 负片: 高光颗粒更多
        "positive": (1.2, 1.0, 0.6, 0.10),   # 正片: 暗部颗粒更多
        "color":    (1.0, 1.0, 0.8, 0.20),   # 彩色负片
        "bw":       (1.1, 1.0, 0.7, 0.0),    # 黑白负片
    }

    def __init__(
        self,
        iso: int = 400,
        film_type: str = "color",
        seed: int = None
    ):
        """
        初始化胶片颗粒生成器

        Args:
            iso: ISO 感光度 (50-3200)
            film_type: 胶片类型 ("negative", "positive", "color", "bw")
            seed: 随机种子
        """
        self.iso = self._find_nearest_iso(iso)
        self.film_type = film_type if film_type in self.FILM_TYPES else "color"
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.perlin = PerlinNoise(seed)

        # 获取预设参数
        self.intensity, self.small_ratio, self.large_ratio, self.cluster_strength = \
            self.ISO_PRESETS[self.iso]
        self.shadow_resp, self.midtone_resp, self.highlight_resp, self.chroma_var = \
            self.FILM_TYPES[self.film_type]

    def _find_nearest_iso(self, iso: int) -> int:
        """找到最近的 ISO 预设值"""
        valid_isos = sorted(self.ISO_PRESETS.keys())
        return min(valid_isos, key=lambda x: abs(x - iso))

    def _compute_grain_sizes(self, height: int, width: int) -> tuple:
        """根据图像分辨率计算实际颗粒尺寸"""
        # 使用短边作为基准
        base_size = min(height, width)

        # 计算实际像素尺寸 (至少 1 像素)
        small_size = max(1, int(base_size * self.small_ratio))
        large_size = max(1, int(base_size * self.large_ratio))

        return small_size, large_size

    def _generate_bimodal_grain(self, height: int, width: int) -> np.ndarray:
        """
        生成双峰分布颗粒 (分辨率自适应)

        70% 小颗粒 (精细纹理) + 30% 大颗粒 (粗糙纹理)
        颗粒尺寸根据图像分辨率自动调整
        """
        # 根据图像分辨率计算颗粒尺寸
        small_scale, large_scale = self._compute_grain_sizes(height, width)

        # 小颗粒层 (70%)
        small_h = max(1, (height + small_scale - 1) // small_scale)
        small_w = max(1, (width + small_scale - 1) // small_scale)
        small_noise = self.rng.standard_normal((small_h, small_w))

        if small_scale > 1:
            small_noise = np.repeat(np.repeat(small_noise, small_scale, axis=0), small_scale, axis=1)
        small_grain = small_noise[:height, :width] * 0.7

        # 大颗粒层 (30%)
        large_h = max(1, (height + large_scale - 1) // large_scale)
        large_w = max(1, (width + large_scale - 1) // large_scale)
        large_noise = self.rng.standard_normal((large_h, large_w))

        if large_scale > 1:
            large_noise = np.repeat(np.repeat(large_noise, large_scale, axis=0), large_scale, axis=1)
        large_grain = large_noise[:height, :width] * 0.3

        return small_grain + large_grain

    def _generate_clustered_grain(self, height: int, width: int) -> np.ndarray:
        """
        使用 Perlin 噪声生成聚簇颗粒 (分辨率自适应)

        真实胶片中，银盐晶体会形成小群落而非均匀分布
        聚簇尺寸根据图像分辨率自动调整
        """
        # 基础颗粒
        base_grain = self._generate_bimodal_grain(height, width)

        # 聚簇尺寸按分辨率缩放
        base_size = min(height, width)
        # 聚簇大小为图像短边的 1.5%-3% (根据 ISO 调整)
        cluster_ratio = 0.015 + (self.iso / 3200) * 0.015
        cluster_scale = max(10, base_size * cluster_ratio)

        cluster_map = self.perlin.fractal_noise(
            width, height,
            scale=cluster_scale,
            octaves=4,
            persistence=0.5
        )

        # 归一化聚簇图到 [0.5, 1.5] 范围
        cluster_map = 0.5 + (cluster_map - cluster_map.min()) / \
                      (cluster_map.max() - cluster_map.min() + 1e-8)

        # 混合: 部分聚簇 + 部分随机
        mixed_grain = base_grain * (
            (1 - self.cluster_strength) +
            self.cluster_strength * cluster_map
        )

        return mixed_grain

    def _compute_luminance_response(self, image: np.ndarray) -> np.ndarray:
        """
        计算分区亮度响应

        根据 Dehancer 研究:
        - 负片: 高光区域颗粒更明显
        - 正片: 暗部区域颗粒更明显
        """
        # 计算亮度
        luminance = (
            0.299 * image[:, :, 0] +
            0.587 * image[:, :, 1] +
            0.114 * image[:, :, 2]
        ) / 255.0

        # 分区响应
        response = np.zeros_like(luminance)

        # 暗部 (0 - 0.3)
        shadow_mask = luminance < 0.3
        response[shadow_mask] = self.shadow_resp * (1 - luminance[shadow_mask] / 0.3 * 0.3)

        # 中间调 (0.3 - 0.7)
        midtone_mask = (luminance >= 0.3) & (luminance < 0.7)
        response[midtone_mask] = self.midtone_resp

        # 高光 (0.7 - 1.0)
        highlight_mask = luminance >= 0.7
        response[highlight_mask] = self.highlight_resp * (1 - (luminance[highlight_mask] - 0.7) / 0.3 * 0.5)

        # 平滑过渡
        response = np.clip(response, 0.2, 1.5)

        return response[:, :, np.newaxis]

    def _generate_color_grain(self, height: int, width: int) -> np.ndarray:
        """
        生成带空间相关性的彩色颗粒 (分辨率自适应)

        真实彩色胶片的染料云 (dye clouds) 特性:
        - RGB 通道有一定空间相关性 (不是完全独立)
        - 存在色彩偏移变化
        """
        # 基础亮度颗粒 (RGB 共享)
        base_grain = self._generate_clustered_grain(height, width)

        # 各通道独立颗粒 (较弱)
        r_independent = self.rng.standard_normal((height, width)) * 0.3
        g_independent = self.rng.standard_normal((height, width)) * 0.3
        b_independent = self.rng.standard_normal((height, width)) * 0.3

        # 空间相关性混合 (70% 共享 + 30% 独立)
        correlation = 0.7
        r_grain = correlation * base_grain + (1 - correlation) * r_independent
        g_grain = correlation * base_grain + (1 - correlation) * g_independent
        b_grain = correlation * base_grain + (1 - correlation) * b_independent

        # 添加色彩偏移 (chroma variance) - 分辨率自适应
        if self.chroma_var > 0:
            base_size = min(height, width)
            chroma_scale = max(20, base_size * 0.04)  # 短边的 4%
            chroma_noise = self.perlin.fractal_noise(width, height, scale=chroma_scale, octaves=2)
            chroma_noise = (chroma_noise - chroma_noise.mean()) * self.chroma_var

            r_grain += chroma_noise * 0.8
            g_grain -= chroma_noise * 0.3
            b_grain += chroma_noise * 0.5

        return np.stack([r_grain, g_grain, b_grain], axis=-1)

    def _generate_bw_grain(self, height: int, width: int) -> np.ndarray:
        """
        生成黑白胶片颗粒

        黑白胶片特性:
        - 金属银晶体,边缘更锐利
        - 单通道颗粒
        - 更高的微对比度
        """
        grain = self._generate_clustered_grain(height, width)

        # 增加锐度 (模拟金属晶体硬边)
        grain = np.sign(grain) * np.power(np.abs(grain), 0.85)

        # 扩展到三通道
        return np.stack([grain, grain, grain], axis=-1)

    def _apply_softening(self, grain: np.ndarray, image: Image.Image) -> np.ndarray:
        """
        对彩色颗粒应用轻微模糊

        彩色胶片的染料云边缘较软,不像黑白那样锐利
        """
        if self.film_type in ["bw"]:
            return grain  # 黑白不模糊

        # 轻微高斯模糊
        grain_image = Image.fromarray(
            ((grain + 2) / 4 * 255).clip(0, 255).astype(np.uint8)
        )
        grain_image = grain_image.filter(ImageFilter.GaussianBlur(radius=0.5))
        grain = (np.array(grain_image).astype(np.float32) / 255 * 4 - 2)

        return grain

    def apply(
        self,
        image: Image.Image,
        intensity_override: float = None
    ) -> Image.Image:
        """
        对图像应用胶片颗粒效果

        Args:
            image: PIL Image 对象
            intensity_override: 覆盖默认强度 (0.0-1.0)

        Returns:
            处理后的 PIL Image
        """
        # 转换为 numpy 数组
        img_array = np.array(image).astype(np.float32)

        # 处理灰度图
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        # 处理 alpha 通道
        has_alpha = img_array.shape[-1] == 4
        if has_alpha:
            alpha = img_array[:, :, 3:4]
            img_array = img_array[:, :, :3]

        height, width = img_array.shape[:2]

        # 生成颗粒
        if self.film_type == "bw":
            grain = self._generate_bw_grain(height, width)
        else:
            grain = self._generate_color_grain(height, width)
            grain = self._apply_softening(grain, image)

        # 计算亮度响应
        lum_response = self._compute_luminance_response(img_array)

        # 应用强度
        intensity = intensity_override if intensity_override is not None else self.intensity
        grain = grain * intensity * 255.0 * lum_response

        # 混合
        result = img_array + grain
        result = np.clip(result, 0, 255).astype(np.uint8)

        # 恢复 alpha
        if has_alpha:
            result = np.concatenate([result, alpha.astype(np.uint8)], axis=-1)

        return Image.fromarray(result)


def process_image(
    input_path: str,
    output_path: str = None,
    iso: int = 400,
    film_type: str = "color",
    intensity: float = None,
    seed: int = None
) -> str:
    """处理单张图像"""
    image = Image.open(input_path)
    grain = FilmGrain(iso=iso, film_type=film_type, seed=seed)
    result = grain.apply(image, intensity_override=intensity)

    if output_path is None:
        path = Path(input_path)
        output_path = str(path.parent / f"{path.stem}_grain{path.suffix}")

    result.save(output_path, quality=95)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="为图像添加真实胶片颗粒效果 v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s photo.jpg                      # 默认 (ISO 400, 彩色负片)
  %(prog)s photo.jpg --iso 1600           # 高感光度粗颗粒
  %(prog)s photo.jpg --type bw            # 黑白胶片
  %(prog)s photo.jpg --type positive      # 正片 (暗部颗粒更多)
  %(prog)s photo.jpg --intensity 0.08     # 自定义强度

胶片类型:
  color     彩色负片 (默认, 如 Kodak Portra)
  bw        黑白负片 (如 Kodak Tri-X)
  negative  通用负片
  positive  正片/幻灯片 (如 Kodak Ektachrome)
        """
    )

    parser.add_argument("input", help="输入图像路径")
    parser.add_argument("-o", "--output", help="输出图像路径")
    parser.add_argument(
        "--iso", type=int, default=400,
        choices=[50, 100, 200, 400, 800, 1600, 3200],
        help="ISO 感光度 (默认: 400)"
    )
    parser.add_argument(
        "--type", dest="film_type", default="color",
        choices=["color", "bw", "negative", "positive"],
        help="胶片类型 (默认: color)"
    )
    parser.add_argument(
        "--intensity", type=float,
        help="颗粒强度 0.0-0.3 (覆盖 ISO 预设)"
    )
    parser.add_argument(
        "--seed", type=int,
        help="随机种子 (用于可重复结果)"
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"错误: 找不到文件 '{args.input}'")
        return 1

    try:
        output_path = process_image(
            input_path=args.input,
            output_path=args.output,
            iso=args.iso,
            film_type=args.film_type,
            intensity=args.intensity,
            seed=args.seed
        )
        print(f"✅ 已保存: {output_path}")
        print(f"   ISO: {args.iso}")
        print(f"   类型: {args.film_type}")
        return 0
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
