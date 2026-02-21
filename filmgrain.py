#!/usr/bin/env python3
"""
Film Grain Effect Generator
基于胶片颗粒特性研究的图像处理脚本

特性:
- 支持 ISO 感光度模拟 (50-3200)
- 黑白/彩色胶片颗粒差异
- 亮度响应 (暗部颗粒更明显)
- 有机随机分布
"""

import numpy as np
from PIL import Image
import argparse
from pathlib import Path


class FilmGrain:
    """胶片颗粒生成器"""

    # ISO 预设参数 (intensity, size, roughness)
    ISO_PRESETS = {
        50:   (0.02, 1.0, 0.3),
        100:  (0.03, 1.2, 0.4),
        200:  (0.04, 1.4, 0.5),
        400:  (0.06, 1.6, 0.6),
        800:  (0.08, 2.0, 0.7),
        1600: (0.12, 2.5, 0.8),
        3200: (0.18, 3.0, 0.9),
    }

    def __init__(self, iso: int = 400, color_mode: str = "color", seed: int = None):
        """
        初始化胶片颗粒生成器

        Args:
            iso: ISO 感光度 (50-3200)
            color_mode: "color" 彩色胶片 / "bw" 黑白胶片
            seed: 随机种子 (用于可重复结果)
        """
        self.iso = self._clamp_iso(iso)
        self.color_mode = color_mode
        self.rng = np.random.default_rng(seed)

        # 获取 ISO 对应的参数
        self.intensity, self.grain_size, self.roughness = self._get_iso_params()

    def _clamp_iso(self, iso: int) -> int:
        """限制 ISO 到有效范围"""
        valid_isos = sorted(self.ISO_PRESETS.keys())
        if iso <= valid_isos[0]:
            return valid_isos[0]
        if iso >= valid_isos[-1]:
            return valid_isos[-1]
        # 找最近的 ISO 值
        return min(valid_isos, key=lambda x: abs(x - iso))

    def _get_iso_params(self) -> tuple:
        """获取 ISO 对应的颗粒参数"""
        return self.ISO_PRESETS.get(self.iso, self.ISO_PRESETS[400])

    def _generate_base_grain(self, shape: tuple) -> np.ndarray:
        """
        生成基础颗粒噪声
        使用高斯分布 + 尺寸调整实现有机感
        """
        h, w = shape[:2]

        # 根据颗粒尺寸缩小生成噪声，再放大
        scale = max(1, int(self.grain_size))
        small_h, small_w = max(1, h // scale), max(1, w // scale)

        # 生成小尺寸高斯噪声
        noise = self.rng.standard_normal((small_h, small_w))

        # 放大到原尺寸 (产生颗粒块状感)
        if scale > 1:
            noise = np.repeat(np.repeat(noise, scale, axis=0), scale, axis=1)
            noise = noise[:h, :w]

        return noise

    def _generate_color_grain(self, shape: tuple) -> np.ndarray:
        """
        生成彩色胶片颗粒 (染料云效果)
        彩色颗粒具有软边缘和多色偏移
        """
        h, w = shape[:2]

        # RGB 三通道独立颗粒 (模拟染料云)
        r_grain = self._generate_base_grain((h, w)) * 0.8
        g_grain = self._generate_base_grain((h, w)) * 0.9
        b_grain = self._generate_base_grain((h, w)) * 0.85

        # 组合成彩色颗粒
        color_grain = np.stack([r_grain, g_grain, b_grain], axis=-1)

        # 应用软化 (彩色胶片颗粒边缘较软)
        return color_grain * 0.7

    def _generate_bw_grain(self, shape: tuple) -> np.ndarray:
        """
        生成黑白胶片颗粒 (金属银晶体)
        黑白颗粒具有硬边缘和更高锐度
        """
        h, w = shape[:2]

        # 单通道颗粒
        grain = self._generate_base_grain((h, w))

        # 应用 roughness 增加硬边效果
        if self.roughness > 0.5:
            # 轻微量化以模拟晶体硬边
            grain = np.sign(grain) * np.power(np.abs(grain), 0.8)

        # 扩展为三通道
        return np.stack([grain, grain, grain], axis=-1)

    def _compute_luminosity_mask(self, image: np.ndarray) -> np.ndarray:
        """
        计算亮度响应掩码
        暗部颗粒更明显，亮部颗粒较轻
        """
        # 转换到亮度 (使用感知权重)
        if len(image.shape) == 3:
            luminosity = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            luminosity = image

        # 归一化到 0-1
        luminosity = luminosity / 255.0

        # 反转并调整曲线 (暗部 = 高响应)
        # 使用 S 曲线使过渡更自然
        response = 1.0 - np.power(luminosity, 0.6)

        # 确保中间调也有一定颗粒
        response = 0.3 + 0.7 * response

        return response[:, :, np.newaxis]

    def apply(self, image: Image.Image, intensity_override: float = None) -> Image.Image:
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

        # 移除 alpha 通道 (如果存在)
        has_alpha = img_array.shape[-1] == 4
        if has_alpha:
            alpha = img_array[:, :, 3:4]
            img_array = img_array[:, :, :3]

        # 生成颗粒
        if self.color_mode == "bw":
            grain = self._generate_bw_grain(img_array.shape)
        else:
            grain = self._generate_color_grain(img_array.shape)

        # 计算亮度响应掩码
        lum_mask = self._compute_luminosity_mask(img_array)

        # 应用强度
        intensity = intensity_override if intensity_override is not None else self.intensity
        grain = grain * intensity * 255.0 * lum_mask

        # 混合颗粒
        result = img_array + grain

        # 裁剪到有效范围
        result = np.clip(result, 0, 255).astype(np.uint8)

        # 恢复 alpha 通道
        if has_alpha:
            result = np.concatenate([result, alpha.astype(np.uint8)], axis=-1)

        return Image.fromarray(result)


def process_image(
    input_path: str,
    output_path: str = None,
    iso: int = 400,
    color_mode: str = "color",
    intensity: float = None,
    seed: int = None
) -> str:
    """
    处理单张图像

    Args:
        input_path: 输入图像路径
        output_path: 输出路径 (默认为 input_grain.ext)
        iso: ISO 感光度
        color_mode: "color" 或 "bw"
        intensity: 自定义强度 (可选)
        seed: 随机种子 (可选)

    Returns:
        输出文件路径
    """
    # 加载图像
    image = Image.open(input_path)

    # 创建颗粒生成器
    grain = FilmGrain(iso=iso, color_mode=color_mode, seed=seed)

    # 应用效果
    result = grain.apply(image, intensity_override=intensity)

    # 确定输出路径
    if output_path is None:
        path = Path(input_path)
        output_path = str(path.parent / f"{path.stem}_grain{path.suffix}")

    # 保存结果
    result.save(output_path, quality=95)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="为图像添加胶片风格颗粒效果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s photo.jpg                    # 使用默认参数 (ISO 400, 彩色)
  %(prog)s photo.jpg -o output.jpg      # 指定输出路径
  %(prog)s photo.jpg --iso 1600         # 高感光度 (粗颗粒)
  %(prog)s photo.jpg --iso 100          # 低感光度 (细颗粒)
  %(prog)s photo.jpg --bw               # 黑白胶片颗粒
  %(prog)s photo.jpg --intensity 0.15   # 自定义强度
        """
    )

    parser.add_argument("input", help="输入图像路径")
    parser.add_argument("-o", "--output", help="输出图像路径")
    parser.add_argument(
        "--iso",
        type=int,
        default=400,
        choices=[50, 100, 200, 400, 800, 1600, 3200],
        help="ISO 感光度 (默认: 400)"
    )
    parser.add_argument(
        "--bw",
        action="store_true",
        help="使用黑白胶片颗粒 (默认: 彩色)"
    )
    parser.add_argument(
        "--intensity",
        type=float,
        help="颗粒强度 0.0-1.0 (覆盖 ISO 预设)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="随机种子 (用于可重复结果)"
    )

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.input).exists():
        print(f"错误: 找不到文件 '{args.input}'")
        return 1

    # 处理图像
    color_mode = "bw" if args.bw else "color"

    try:
        output_path = process_image(
            input_path=args.input,
            output_path=args.output,
            iso=args.iso,
            color_mode=color_mode,
            intensity=args.intensity,
            seed=args.seed
        )
        print(f"已保存: {output_path}")
        print(f"  ISO: {args.iso}")
        print(f"  模式: {'黑白' if args.bw else '彩色'}胶片")
        if args.intensity:
            print(f"  强度: {args.intensity}")
        return 0
    except Exception as e:
        print(f"处理失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
