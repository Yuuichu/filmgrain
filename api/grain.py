"""
Film Grain API v3.0 - 真实胶片颗粒模拟
基于物理模型: 泊松分布 + Simplex聚簇 + 高斯splat形态
"""

from http.server import BaseHTTPRequestHandler
import json
import base64
import io
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

# 注册 HEIC 支持
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


class SimplexNoise:
    """
    Simplex 噪声 - 比 Perlin 更快、无方向性伪影
    基于 Ken Perlin 的改进算法
    """

    # Simplex 梯度表
    GRAD3 = np.array([
        [1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],
        [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],
        [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]
    ], dtype=np.float32)

    F2 = 0.5 * (np.sqrt(3.0) - 1.0)
    G2 = (3.0 - np.sqrt(3.0)) / 6.0

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        p = np.arange(256, dtype=np.int32)
        self.rng.shuffle(p)
        self.perm = np.concatenate([p, p])
        self.perm_mod12 = self.perm % 12

    def noise2d(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """2D Simplex 噪声"""
        s = (x + y) * self.F2
        i = np.floor(x + s).astype(np.int32)
        j = np.floor(y + s).astype(np.int32)

        t = (i + j) * self.G2
        X0 = i - t
        Y0 = j - t
        x0 = x - X0
        y0 = y - Y0

        i1 = (x0 > y0).astype(np.int32)
        j1 = 1 - i1

        x1 = x0 - i1 + self.G2
        y1 = y0 - j1 + self.G2
        x2 = x0 - 1.0 + 2.0 * self.G2
        y2 = y0 - 1.0 + 2.0 * self.G2

        ii = i & 255
        jj = j & 255

        gi0 = self.perm_mod12[self.perm[ii + self.perm[jj]]]
        gi1 = self.perm_mod12[self.perm[ii + i1 + self.perm[jj + j1]]]
        gi2 = self.perm_mod12[self.perm[ii + 1 + self.perm[jj + 1]]]

        t0 = 0.5 - x0*x0 - y0*y0
        t1 = 0.5 - x1*x1 - y1*y1
        t2 = 0.5 - x2*x2 - y2*y2

        n0 = np.where(t0 < 0, 0, t0**4 * (self.GRAD3[gi0, 0] * x0 + self.GRAD3[gi0, 1] * y0))
        n1 = np.where(t1 < 0, 0, t1**4 * (self.GRAD3[gi1, 0] * x1 + self.GRAD3[gi1, 1] * y1))
        n2 = np.where(t2 < 0, 0, t2**4 * (self.GRAD3[gi2, 0] * x2 + self.GRAD3[gi2, 1] * y2))

        return 70.0 * (n0 + n1 + n2)

    def fractal_noise(self, width: int, height: int, scale: float = 50.0,
                      octaves: int = 4, persistence: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
        """分形 Simplex 噪声"""
        y, x = np.meshgrid(np.arange(height, dtype=np.float32),
                           np.arange(width, dtype=np.float32), indexing='ij')

        noise = np.zeros((height, width), dtype=np.float32)
        amplitude = 1.0
        max_amp = 0.0
        freq = 1.0

        for _ in range(octaves):
            noise += amplitude * self.noise2d(x * freq / scale, y * freq / scale)
            max_amp += amplitude
            amplitude *= persistence
            freq *= lacunarity

        return noise / max_amp


class RealisticFilmGrain:
    """
    真实胶片颗粒生成器 v3.0

    物理模型:
    1. 银盐晶体位置: 调制泊松分布 (Modulated Poisson)
    2. 晶体聚簇: Simplex 噪声密度调制
    3. 晶体形态: 高斯 splat (软边圆形)
    4. 尺寸分布: 对数正态分布 (Log-normal)
    """

    ISO_PRESETS = {
        # (density, mean_size_ratio, size_variance, cluster_strength, softness)
        50:   (0.012, 0.0006, 0.2, 0.15, 0.8),
        100:  (0.020, 0.0008, 0.25, 0.20, 0.75),
        200:  (0.030, 0.0010, 0.30, 0.25, 0.70),
        400:  (0.045, 0.0012, 0.35, 0.30, 0.65),
        800:  (0.065, 0.0015, 0.40, 0.38, 0.55),
        1600: (0.095, 0.0020, 0.45, 0.45, 0.45),
        3200: (0.130, 0.0025, 0.50, 0.55, 0.35),
    }

    FILM_TYPES = {
        "color":    {"shadow": 1.0, "mid": 1.0, "high": 0.75, "chroma": 0.20, "softness_mult": 1.2},
        "bw":       {"shadow": 1.1, "mid": 1.0, "high": 0.65, "chroma": 0.0,  "softness_mult": 0.8},
        "negative": {"shadow": 0.8, "mid": 1.0, "high": 1.15, "chroma": 0.15, "softness_mult": 1.0},
        "positive": {"shadow": 1.2, "mid": 1.0, "high": 0.55, "chroma": 0.10, "softness_mult": 1.1},
    }

    def __init__(self, iso: int = 400, film_type: str = "color", seed: int = None):
        self.iso = min(self.ISO_PRESETS.keys(), key=lambda x: abs(x - iso))
        self.film_type = film_type if film_type in self.FILM_TYPES else "color"
        self.rng = np.random.default_rng(seed)
        self.simplex = SimplexNoise(seed)

        preset = self.ISO_PRESETS[self.iso]
        self.density = preset[0]
        self.mean_size_ratio = preset[1]
        self.size_variance = preset[2]
        self.cluster_strength = preset[3]
        self.base_softness = preset[4]

        film = self.FILM_TYPES[self.film_type]
        self.shadow_resp = film["shadow"]
        self.mid_resp = film["mid"]
        self.high_resp = film["high"]
        self.chroma = film["chroma"]
        self.softness = self.base_softness * film["softness_mult"]

    def _generate_grain_field(self, height: int, width: int) -> np.ndarray:
        """
        生成颗粒场 - 使用调制泊松分布

        真实胶片: 银盐晶体呈泊松分布，但密度受局部条件调制
        """
        base_size = min(height, width)

        # 1. 生成基础高斯噪声场
        noise = self.rng.standard_normal((height, width)).astype(np.float32)

        # 2. 生成聚簇密度图 (Simplex)
        cluster_scale = base_size * (0.02 + self.iso / 8000)
        cluster_map = self.simplex.fractal_noise(
            width, height,
            scale=cluster_scale,
            octaves=5,
            persistence=0.6,
            lacunarity=2.0
        )
        # 归一化到 [0.3, 1.7]
        cluster_map = (cluster_map - cluster_map.min()) / (cluster_map.max() - cluster_map.min() + 1e-8)
        cluster_map = 0.3 + cluster_map * 1.4

        # 3. 调制噪声密度
        modulated = noise * ((1 - self.cluster_strength) + self.cluster_strength * cluster_map)

        return modulated

    def _apply_grain_shape(self, grain: np.ndarray, height: int, width: int) -> np.ndarray:
        """
        应用颗粒形态 - 高斯 splat

        真实胶片颗粒是软边圆形/椭圆形，不是硬边方块
        使用高斯模糊 + 对比度增强模拟
        """
        base_size = min(height, width)
        grain_size = max(1, int(base_size * self.mean_size_ratio))

        # 高斯模糊模拟软边颗粒
        sigma = grain_size * self.softness * 0.5
        if sigma > 0.3:
            grain = ndimage.gaussian_filter(grain, sigma=sigma)

            # 对比度补偿 (模糊会降低对比度)
            grain = grain * (1 + sigma * 0.3)

        return grain

    def _apply_size_distribution(self, grain: np.ndarray, height: int, width: int) -> np.ndarray:
        """
        应用尺寸分布 - 对数正态混合

        真实胶片: 晶体尺寸呈对数正态分布
        大晶体稀疏但明显，小晶体密集但细微
        """
        base_size = min(height, width)

        # 生成多尺度颗粒并混合
        scales = [0.5, 1.0, 2.0]  # 小/中/大晶体
        weights = [0.25, 0.50, 0.25]  # 对数正态近似

        result = np.zeros_like(grain)
        for scale, weight in zip(scales, weights):
            grain_size = max(1, int(base_size * self.mean_size_ratio * scale))
            if grain_size > 1:
                # 下采样再上采样产生不同尺度
                small_h = max(1, height // grain_size)
                small_w = max(1, width // grain_size)
                small = self.rng.standard_normal((small_h, small_w)).astype(np.float32)

                # 双三次插值上采样 (更平滑)
                from PIL import Image as PILImage
                small_img = PILImage.fromarray(small, mode='F')
                large_img = small_img.resize((width, height), PILImage.Resampling.BICUBIC)
                layer = np.array(large_img)
            else:
                layer = self.rng.standard_normal((height, width)).astype(np.float32)

            result += layer * weight

        # 与原始颗粒场混合
        return grain * 0.4 + result * 0.6

    def _compute_luminance_response(self, image: np.ndarray) -> np.ndarray:
        """
        计算亮度响应曲线

        使用平滑 S 曲线插值，避免硬边界
        """
        lum = (0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]) / 255.0

        # 平滑三区响应
        response = np.zeros_like(lum)

        # 使用 smoothstep 插值
        def smoothstep(edge0, edge1, x):
            t = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
            return t * t * (3 - 2 * t)

        # 暗部到中间调过渡 (0-0.35)
        shadow_blend = smoothstep(0, 0.35, lum)
        # 中间调到高光过渡 (0.55-1.0)
        highlight_blend = smoothstep(0.55, 1.0, lum)

        response = (
            self.shadow_resp * (1 - shadow_blend) +
            self.mid_resp * shadow_blend * (1 - highlight_blend) +
            self.high_resp * highlight_blend
        )

        return np.clip(response, 0.15, 1.5)[:, :, np.newaxis]

    def _generate_color_grain(self, height: int, width: int) -> np.ndarray:
        """
        生成彩色颗粒 - 染料云模型

        彩色胶片: RGB 通道有空间相关性 + 色彩漂移
        """
        # 基础亮度颗粒
        luma_grain = self._generate_grain_field(height, width)
        luma_grain = self._apply_size_distribution(luma_grain, height, width)

        if self.chroma == 0:
            # 黑白模式
            grain = np.stack([luma_grain] * 3, axis=-1)
        else:
            # 彩色模式: 各通道部分独立
            correlation = 0.75  # 通道相关性

            r_ind = self._generate_grain_field(height, width) * 0.25
            g_ind = self._generate_grain_field(height, width) * 0.25
            b_ind = self._generate_grain_field(height, width) * 0.25

            r = correlation * luma_grain + (1 - correlation) * r_ind
            g = correlation * luma_grain + (1 - correlation) * g_ind
            b = correlation * luma_grain + (1 - correlation) * b_ind

            # 色彩漂移 (染料云特性)
            if self.chroma > 0:
                base_size = min(height, width)
                chroma_scale = base_size * 0.05
                chroma_noise = self.simplex.fractal_noise(
                    width, height, scale=chroma_scale, octaves=3
                )
                chroma_noise = (chroma_noise - chroma_noise.mean()) * self.chroma

                r += chroma_noise * 0.9
                g -= chroma_noise * 0.4
                b += chroma_noise * 0.6

            grain = np.stack([r, g, b], axis=-1)

        return grain

    def apply(self, image: Image.Image, intensity_override: float = None) -> Image.Image:
        """应用胶片颗粒效果"""
        img_array = np.array(image).astype(np.float32)

        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        has_alpha = img_array.shape[-1] == 4
        if has_alpha:
            alpha = img_array[:, :, 3:4]
            img_array = img_array[:, :, :3]

        height, width = img_array.shape[:2]

        # 生成颗粒
        grain = self._generate_color_grain(height, width)

        # 应用颗粒形态 (各通道)
        for c in range(3):
            grain[:,:,c] = self._apply_grain_shape(grain[:,:,c], height, width)

        # 亮度响应
        lum_response = self._compute_luminance_response(img_array)

        # 强度
        intensity = intensity_override if intensity_override is not None else self.density
        grain = grain * intensity * 255.0 * lum_response

        # 混合
        result = np.clip(img_array + grain, 0, 255).astype(np.uint8)

        if has_alpha:
            result = np.concatenate([result, alpha.astype(np.uint8)], axis=-1)

        return Image.fromarray(result)


# API Handler
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        info = {
            "name": "Film Grain API",
            "version": "3.0",
            "model": "Physical simulation",
            "features": [
                "Simplex 噪声聚簇 (无方向伪影)",
                "对数正态尺寸分布",
                "高斯 splat 颗粒形态",
                "平滑 S 曲线亮度响应",
                "RGB 空间相关性 + 色彩漂移"
            ],
            "parameters": {
                "image": "base64 encoded (required)",
                "iso": "50-3200 (default: 400)",
                "type": "color, bw, negative, positive",
                "intensity": "0.01-0.3 (optional)"
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
            film_type = data.get('type', data.get('mode', 'color'))
            intensity = data.get('intensity')

            if intensity:
                intensity = float(intensity)

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

            grain = RealisticFilmGrain(iso=iso, film_type=film_type)
            result = grain.apply(image, intensity_override=intensity)

            buffer = io.BytesIO()
            result.save(buffer, format='JPEG', quality=92)
            result_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            response = {
                "success": True,
                "image": result_b64,
                "params": {"iso": iso, "type": film_type, "version": "3.0"}
            }
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
