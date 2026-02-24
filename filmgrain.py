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
    """分形噪声生成器 - 用于有机聚簇效果"""

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def fractal_noise(self, width: int, height: int, scale: float = 50.0,
                      octaves: int = 4, persistence: float = 0.5) -> np.ndarray:
        """
        生成分形噪声 (多层平滑噪声叠加)

        使用双线性插值实现平滑的有机聚簇效果

        Args:
            width, height: 输出尺寸
            scale: 基础缩放 (越大 = 聚簇越大)
            octaves: 叠加层数
            persistence: 高频衰减系数
        """
        noise = np.zeros((height, width), dtype=np.float32)
        amplitude = 1.0
        max_amplitude = 0.0
        current_scale = scale

        for _ in range(octaves):
            # 生成低分辨率噪声
            small_h = max(2, int(height / current_scale))
            small_w = max(2, int(width / current_scale))
            small_noise = self.rng.standard_normal((small_h, small_w)).astype(np.float32)

            # 使用双线性插值放大到目标尺寸 (产生平滑过渡)
            small_img = Image.fromarray(small_noise, mode='F')
            large_img = small_img.resize((width, height), Image.Resampling.BILINEAR)
            layer = np.array(large_img)

            noise += amplitude * layer
            max_amplitude += amplitude
            amplitude *= persistence
            current_scale /= 2

        return noise / max_amplitude


class FilmGrain:
    """
    真实胶片颗粒生成器 v2.0

    基于 Dehancer 算法研究和 City Frame Photography 技术文档
    """

    # ISO 预设: (base_intensity, small_grain_size, large_grain_size, cluster_strength)
    ISO_PRESETS = {
        50:   (0.015, 0.8, 1.5, 0.2),   # 超细腻
        100:  (0.025, 1.0, 2.0, 0.25),  # 细腻
        200:  (0.035, 1.2, 2.5, 0.3),   # 细腻-中等
        400:  (0.050, 1.5, 3.0, 0.4),   # 标准
        800:  (0.070, 2.0, 4.0, 0.5),   # 明显
        1600: (0.100, 2.5, 5.0, 0.6),   # 粗颗粒
        3200: (0.140, 3.0, 6.0, 0.7),   # 超粗颗粒
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
        self.intensity, self.small_size, self.large_size, self.cluster_strength = \
            self.ISO_PRESETS[self.iso]
        self.shadow_resp, self.midtone_resp, self.highlight_resp, self.chroma_var = \
            self.FILM_TYPES[self.film_type]

    def _find_nearest_iso(self, iso: int) -> int:
        """找到最近的 ISO 预设值"""
        valid_isos = sorted(self.ISO_PRESETS.keys())
        return min(valid_isos, key=lambda x: abs(x - iso))

    def _generate_bimodal_grain(self, height: int, width: int) -> np.ndarray:
        """
        生成双峰分布颗粒

        70% 小颗粒 (精细纹理) + 30% 大颗粒 (粗糙纹理)
        这模拟了真实胶片中混合尺寸的银盐晶体
        """
        # 小颗粒层 (70%) - 使用 ceil 确保放大后尺寸足够
        small_scale = max(1, int(self.small_size))
        small_h = max(1, (height + small_scale - 1) // small_scale)
        small_w = max(1, (width + small_scale - 1) // small_scale)
        small_noise = self.rng.standard_normal((small_h, small_w))

        if small_scale > 1:
            small_noise = np.repeat(np.repeat(small_noise, small_scale, axis=0), small_scale, axis=1)
        small_grain = small_noise[:height, :width] * 0.7

        # 大颗粒层 (30%)
        large_scale = max(1, int(self.large_size))
        large_h = max(1, (height + large_scale - 1) // large_scale)
        large_w = max(1, (width + large_scale - 1) // large_scale)
        large_noise = self.rng.standard_normal((large_h, large_w))

        if large_scale > 1:
            large_noise = np.repeat(np.repeat(large_noise, large_scale, axis=0), large_scale, axis=1)
        large_grain = large_noise[:height, :width] * 0.3

        return small_grain + large_grain

    def _generate_clustered_grain(self, height: int, width: int) -> np.ndarray:
        """
        使用 Perlin 噪声生成聚簇颗粒

        真实胶片中，银盐晶体会形成小群落而非均匀分布
        """
        # 基础颗粒
        base_grain = self._generate_bimodal_grain(height, width)

        # Perlin 聚簇调制
        cluster_scale = 30 + self.iso / 50  # ISO 越高，聚簇越大
        cluster_map = self.perlin.fractal_noise(
            width, height,
            scale=cluster_scale,
            octaves=3,
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
        生成带空间相关性的彩色颗粒

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

        # 添加色彩偏移 (chroma variance)
        if self.chroma_var > 0:
            chroma_noise = self.perlin.fractal_noise(width, height, scale=80, octaves=2)
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
