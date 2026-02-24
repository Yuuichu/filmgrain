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
    """Perlin 噪声生成器"""

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.perm = np.arange(256, dtype=np.int32)
        self.rng.shuffle(self.perm)
        self.perm = np.stack([self.perm, self.perm]).flatten()

    def _fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a, b, t):
        return a + t * (b - a)

    def _grad(self, hash_val, x, y):
        h = hash_val & 3
        grad_vectors = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        g = grad_vectors[h]
        return g[0] * x + g[1] * y

    def noise2d(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xi = x.astype(np.int32) & 255
        yi = y.astype(np.int32) & 255
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        u = self._fade(xf)
        v = self._fade(yf)

        aa = self.perm[self.perm[xi] + yi]
        ab = self.perm[self.perm[xi] + yi + 1]
        ba = self.perm[self.perm[xi + 1] + yi]
        bb = self.perm[self.perm[xi + 1] + yi + 1]

        x1 = self._lerp(self._grad(aa, xf, yf), self._grad(ba, xf - 1, yf), u)
        x2 = self._lerp(self._grad(ab, xf, yf - 1), self._grad(bb, xf - 1, yf - 1), u)
        return self._lerp(x1, x2, v)

    def fractal_noise(self, width: int, height: int, scale: float = 50.0,
                      octaves: int = 3, persistence: float = 0.5) -> np.ndarray:
        y_coords, x_coords = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing='ij'
        )
        noise = np.zeros((height, width), dtype=np.float32)
        amplitude = 1.0
        max_amplitude = 0.0

        for _ in range(octaves):
            noise += amplitude * self.noise2d(x_coords / scale, y_coords / scale)
            max_amplitude += amplitude
            amplitude *= persistence
            scale /= 2

        return noise / max_amplitude


class FilmGrain:
    """真实胶片颗粒生成器 v2.0"""

    ISO_PRESETS = {
        50:   (0.015, 0.8, 1.5, 0.2),
        100:  (0.025, 1.0, 2.0, 0.25),
        200:  (0.035, 1.2, 2.5, 0.3),
        400:  (0.050, 1.5, 3.0, 0.4),
        800:  (0.070, 2.0, 4.0, 0.5),
        1600: (0.100, 2.5, 5.0, 0.6),
        3200: (0.140, 3.0, 6.0, 0.7),
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

        self.intensity, self.small_size, self.large_size, self.cluster_strength = \
            self.ISO_PRESETS[self.iso]
        self.shadow_resp, self.midtone_resp, self.highlight_resp, self.chroma_var = \
            self.FILM_TYPES[self.film_type]

    def _generate_bimodal_grain(self, height: int, width: int) -> np.ndarray:
        """双峰分布颗粒: 70% 小 + 30% 大"""
        # 小颗粒
        small_scale = max(1, int(self.small_size))
        small_h, small_w = max(1, height // small_scale), max(1, width // small_scale)
        small_noise = self.rng.standard_normal((small_h, small_w))
        if small_scale > 1:
            small_noise = np.repeat(np.repeat(small_noise, small_scale, axis=0), small_scale, axis=1)
        small_grain = small_noise[:height, :width] * 0.7

        # 大颗粒
        large_scale = max(1, int(self.large_size))
        large_h, large_w = max(1, height // large_scale), max(1, width // large_scale)
        large_noise = self.rng.standard_normal((large_h, large_w))
        if large_scale > 1:
            large_noise = np.repeat(np.repeat(large_noise, large_scale, axis=0), large_scale, axis=1)
        large_grain = large_noise[:height, :width] * 0.3

        return small_grain + large_grain

    def _generate_clustered_grain(self, height: int, width: int) -> np.ndarray:
        """Perlin 噪声聚簇颗粒"""
        base_grain = self._generate_bimodal_grain(height, width)

        cluster_scale = 30 + self.iso / 50
        cluster_map = self.perlin.fractal_noise(width, height, scale=cluster_scale, octaves=3)
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
        """彩色颗粒 (RGB 空间相关性)"""
        base_grain = self._generate_clustered_grain(height, width)

        r_ind = self.rng.standard_normal((height, width)) * 0.3
        g_ind = self.rng.standard_normal((height, width)) * 0.3
        b_ind = self.rng.standard_normal((height, width)) * 0.3

        correlation = 0.7
        r_grain = correlation * base_grain + (1 - correlation) * r_ind
        g_grain = correlation * base_grain + (1 - correlation) * g_ind
        b_grain = correlation * base_grain + (1 - correlation) * b_ind

        if self.chroma_var > 0:
            chroma_noise = self.perlin.fractal_noise(width, height, scale=80, octaves=2)
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
