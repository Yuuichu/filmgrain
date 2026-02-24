"""
Film Grain API v2.0 - Vercel Serverless Function
真实胶片颗粒模拟 API
"""

from http.server import BaseHTTPRequestHandler
import json
import base64
import io
import numpy as np
from PIL import Image, ImageFilter


class PerlinNoise:
    """Perlin 噪声生成器 - 真正的梯度噪声实现"""

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        # 生成排列表
        p = np.arange(256, dtype=np.int32)
        self.rng.shuffle(p)
        self.perm = np.concatenate([p, p])  # 重复以避免溢出

        # 预计算梯度向量
        self.gradients = np.array([
            [1, 1], [-1, 1], [1, -1], [-1, -1],
            [1, 0], [-1, 0], [0, 1], [0, -1]
        ], dtype=np.float32)

    def _fade(self, t):
        """平滑插值曲线 6t^5 - 15t^4 + 10t^3"""
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
        """生成分形噪声 (多层 Perlin 叠加)"""
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
    """真实胶片颗粒生成器 v2.1 - 分辨率自适应"""

    # 基准分辨率 (2K)
    REFERENCE_SIZE = 2048

    # ISO 预设: (intensity, small_grain_ratio, large_grain_ratio, cluster_strength)
    # grain_ratio 是相对于图像短边的比例 (例如 0.001 = 短边的 0.1%)
    ISO_PRESETS = {
        50:   (0.015, 0.0004, 0.0008, 0.2),   # 超细腻
        100:  (0.025, 0.0005, 0.0010, 0.25),
        200:  (0.035, 0.0006, 0.0013, 0.3),
        400:  (0.050, 0.0008, 0.0016, 0.4),   # 标准
        800:  (0.070, 0.0010, 0.0020, 0.5),
        1600: (0.100, 0.0013, 0.0026, 0.6),
        3200: (0.140, 0.0016, 0.0032, 0.7),   # 粗颗粒
    }

    FILM_TYPES = {
        "negative": (0.8, 1.0, 1.2, 0.15),
        "positive": (1.2, 1.0, 0.6, 0.10),
        "color":    (1.0, 1.0, 0.8, 0.20),
        "bw":       (1.1, 1.0, 0.7, 0.0),
    }

    def __init__(self, iso: int = 400, film_type: str = "color", seed: int = None):
        self.iso = min(self.ISO_PRESETS.keys(), key=lambda x: abs(x - iso))
        self.film_type = film_type if film_type in self.FILM_TYPES else "color"
        self.rng = np.random.default_rng(seed)
        self.perlin = PerlinNoise(seed)

        self.intensity, self.small_ratio, self.large_ratio, self.cluster_strength = \
            self.ISO_PRESETS[self.iso]
        self.shadow_resp, self.midtone_resp, self.highlight_resp, self.chroma_var = \
            self.FILM_TYPES[self.film_type]

    def _compute_grain_sizes(self, height: int, width: int) -> tuple:
        """根据图像分辨率计算实际颗粒尺寸"""
        # 使用短边作为基准
        base_size = min(height, width)

        # 计算实际像素尺寸 (至少 1 像素)
        small_size = max(1, int(base_size * self.small_ratio))
        large_size = max(1, int(base_size * self.large_ratio))

        return small_size, large_size

    def _generate_bimodal_grain(self, height: int, width: int) -> np.ndarray:
        """双峰分布颗粒: 70% 小 + 30% 大 (分辨率自适应)"""
        # 根据图像分辨率计算颗粒尺寸
        small_scale, large_scale = self._compute_grain_sizes(height, width)

        # 小颗粒 (70%)
        small_h = max(1, (height + small_scale - 1) // small_scale)
        small_w = max(1, (width + small_scale - 1) // small_scale)
        small_noise = self.rng.standard_normal((small_h, small_w))
        if small_scale > 1:
            small_noise = np.repeat(np.repeat(small_noise, small_scale, axis=0), small_scale, axis=1)
        small_grain = small_noise[:height, :width] * 0.7

        # 大颗粒 (30%)
        large_h = max(1, (height + large_scale - 1) // large_scale)
        large_w = max(1, (width + large_scale - 1) // large_scale)
        large_noise = self.rng.standard_normal((large_h, large_w))
        if large_scale > 1:
            large_noise = np.repeat(np.repeat(large_noise, large_scale, axis=0), large_scale, axis=1)
        large_grain = large_noise[:height, :width] * 0.3

        return small_grain + large_grain

    def _generate_clustered_grain(self, height: int, width: int) -> np.ndarray:
        """Perlin 噪声聚簇颗粒 (分辨率自适应)"""
        base_grain = self._generate_bimodal_grain(height, width)

        # 聚簇尺寸也按分辨率缩放
        base_size = min(height, width)
        # 聚簇大小为图像短边的 1.5%-3% (根据 ISO 调整)
        cluster_ratio = 0.015 + (self.iso / 3200) * 0.015
        cluster_scale = max(10, base_size * cluster_ratio)

        cluster_map = self.perlin.fractal_noise(width, height, scale=cluster_scale, octaves=4)
        cluster_map = 0.5 + (cluster_map - cluster_map.min()) / (cluster_map.max() - cluster_map.min() + 1e-8)

        return base_grain * ((1 - self.cluster_strength) + self.cluster_strength * cluster_map)

    def _compute_luminance_response(self, image: np.ndarray) -> np.ndarray:
        """分区亮度响应"""
        luminance = (0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]) / 255.0
        response = np.zeros_like(luminance)

        shadow_mask = luminance < 0.3
        response[shadow_mask] = self.shadow_resp * (1 - luminance[shadow_mask] / 0.3 * 0.3)

        midtone_mask = (luminance >= 0.3) & (luminance < 0.7)
        response[midtone_mask] = self.midtone_resp

        highlight_mask = luminance >= 0.7
        response[highlight_mask] = self.highlight_resp * (1 - (luminance[highlight_mask] - 0.7) / 0.3 * 0.5)

        return np.clip(response, 0.2, 1.5)[:, :, np.newaxis]

    def _generate_color_grain(self, height: int, width: int) -> np.ndarray:
        """彩色颗粒 (RGB 空间相关性，分辨率自适应)"""
        base_grain = self._generate_clustered_grain(height, width)

        r_ind = self.rng.standard_normal((height, width)) * 0.3
        g_ind = self.rng.standard_normal((height, width)) * 0.3
        b_ind = self.rng.standard_normal((height, width)) * 0.3

        correlation = 0.7
        r_grain = correlation * base_grain + (1 - correlation) * r_ind
        g_grain = correlation * base_grain + (1 - correlation) * g_ind
        b_grain = correlation * base_grain + (1 - correlation) * b_ind

        if self.chroma_var > 0:
            # 色彩偏移也按分辨率缩放
            base_size = min(height, width)
            chroma_scale = max(20, base_size * 0.04)  # 短边的 4%
            chroma_noise = self.perlin.fractal_noise(width, height, scale=chroma_scale, octaves=2)
            chroma_noise = (chroma_noise - chroma_noise.mean()) * self.chroma_var
            r_grain += chroma_noise * 0.8
            g_grain -= chroma_noise * 0.3
            b_grain += chroma_noise * 0.5

        return np.stack([r_grain, g_grain, b_grain], axis=-1)

    def _generate_bw_grain(self, height: int, width: int) -> np.ndarray:
        """黑白颗粒 (硬边晶体)"""
        grain = self._generate_clustered_grain(height, width)
        grain = np.sign(grain) * np.power(np.abs(grain), 0.85)
        return np.stack([grain, grain, grain], axis=-1)

    def apply(self, image: Image.Image, intensity_override: float = None) -> Image.Image:
        img_array = np.array(image).astype(np.float32)

        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        has_alpha = img_array.shape[-1] == 4
        if has_alpha:
            alpha = img_array[:, :, 3:4]
            img_array = img_array[:, :, :3]

        height, width = img_array.shape[:2]

        if self.film_type == "bw":
            grain = self._generate_bw_grain(height, width)
        else:
            grain = self._generate_color_grain(height, width)

        lum_response = self._compute_luminance_response(img_array)
        intensity = intensity_override if intensity_override is not None else self.intensity
        grain = grain * intensity * 255.0 * lum_response

        result = np.clip(img_array + grain, 0, 255).astype(np.uint8)

        if has_alpha:
            result = np.concatenate([result, alpha.astype(np.uint8)], axis=-1)

        return Image.fromarray(result)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        info = {
            "name": "Film Grain API",
            "version": "2.0",
            "features": [
                "双峰尺寸分布 (70% 小颗粒 + 30% 大颗粒)",
                "Perlin 噪声聚簇",
                "分区亮度响应 (暗部/中间调/高光)",
                "RGB 通道空间相关性"
            ],
            "parameters": {
                "image": "base64 encoded image (required)",
                "iso": "50, 100, 200, 400, 800, 1600, 3200 (default: 400)",
                "type": "color, bw, negative, positive (default: color)",
                "intensity": "0.01-0.3 (optional, overrides ISO preset)"
            }
        }
        self.wfile.write(json.dumps(info, ensure_ascii=False).encode())

    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            image_b64 = data.get('image', '')
            iso = int(data.get('iso', 400))
            film_type = data.get('type', data.get('mode', 'color'))  # 兼容旧参数
            intensity = data.get('intensity')

            if intensity:
                intensity = float(intensity)

            # 兼容旧 API: mode=bw -> type=bw
            if film_type == 'bw' or data.get('mode') == 'bw':
                film_type = 'bw'

            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')

            # 限制尺寸
            max_size = 2048
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            grain = FilmGrain(iso=iso, film_type=film_type)
            result = grain.apply(image, intensity_override=intensity)

            buffer = io.BytesIO()
            result.save(buffer, format='JPEG', quality=90)
            result_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            response = {
                "success": True,
                "image": result_b64,
                "params": {
                    "iso": iso,
                    "type": film_type,
                    "intensity": intensity or "auto",
                    "version": "2.0"
                }
            }
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            error_msg = str(e) if str(e) else type(e).__name__
            self.wfile.write(json.dumps({"success": False, "error": error_msg}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
