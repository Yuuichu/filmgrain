"""
Film Grain API - Vercel Serverless Function
部署后可被 iOS 快捷指令调用
"""

from http.server import BaseHTTPRequestHandler
import json
import base64
import io
import numpy as np
from PIL import Image


class FilmGrain:
    """胶片颗粒生成器 (简化版)"""

    ISO_PRESETS = {
        50:   (0.02, 1.0),
        100:  (0.03, 1.2),
        200:  (0.04, 1.4),
        400:  (0.06, 1.6),
        800:  (0.08, 2.0),
        1600: (0.12, 2.5),
        3200: (0.18, 3.0),
    }

    def __init__(self, iso: int = 400, color_mode: str = "color", seed: int = None):
        self.iso = min(self.ISO_PRESETS.keys(), key=lambda x: abs(x - iso))
        self.color_mode = color_mode
        self.rng = np.random.default_rng(seed)
        self.intensity, self.grain_size = self.ISO_PRESETS[self.iso]

    def apply(self, image: Image.Image, intensity_override: float = None) -> Image.Image:
        img_array = np.array(image).astype(np.float32)

        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        has_alpha = img_array.shape[-1] == 4
        if has_alpha:
            alpha = img_array[:, :, 3:4]
            img_array = img_array[:, :, :3]

        h, w = img_array.shape[:2]
        scale = max(1, int(self.grain_size))
        small_h, small_w = max(1, h // scale), max(1, w // scale)

        if self.color_mode == "bw":
            noise = self.rng.standard_normal((small_h, small_w))
            if scale > 1:
                noise = np.repeat(np.repeat(noise, scale, axis=0), scale, axis=1)[:h, :w]
            grain = np.stack([noise, noise, noise], axis=-1)
        else:
            grain = np.stack([
                self._upscale_noise(self.rng.standard_normal((small_h, small_w)), scale, h, w) * 0.8,
                self._upscale_noise(self.rng.standard_normal((small_h, small_w)), scale, h, w) * 0.9,
                self._upscale_noise(self.rng.standard_normal((small_h, small_w)), scale, h, w) * 0.85,
            ], axis=-1) * 0.7

        # 亮度响应
        lum = (0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]) / 255.0
        response = (0.3 + 0.7 * (1.0 - np.power(lum, 0.6)))[:, :, np.newaxis]

        intensity = intensity_override if intensity_override else self.intensity
        result = img_array + grain * intensity * 255.0 * response
        result = np.clip(result, 0, 255).astype(np.uint8)

        if has_alpha:
            result = np.concatenate([result, alpha.astype(np.uint8)], axis=-1)

        return Image.fromarray(result)

    def _upscale_noise(self, noise, scale, h, w):
        if scale > 1:
            noise = np.repeat(np.repeat(noise, scale, axis=0), scale, axis=1)
        return noise[:h, :w]


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """返回 API 信息"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        info = {
            "name": "Film Grain API",
            "version": "1.0",
            "usage": "POST with base64 image",
            "parameters": {
                "image": "base64 encoded image (required)",
                "iso": "50-3200 (default: 400)",
                "mode": "color or bw (default: color)",
                "intensity": "0.01-0.5 (optional, overrides ISO preset)"
            }
        }
        self.wfile.write(json.dumps(info, ensure_ascii=False).encode())

    def do_POST(self):
        """处理图像"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            # 解析参数
            image_b64 = data.get('image', '')
            iso = int(data.get('iso', 400))
            mode = data.get('mode', 'color')
            intensity = data.get('intensity')

            if intensity:
                intensity = float(intensity)

            # 解码图像
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')

            # 限制尺寸 (Vercel 内存限制)
            max_size = 2048
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # 应用效果
            grain = FilmGrain(iso=iso, color_mode=mode)
            result = grain.apply(image, intensity_override=intensity)

            # 编码输出
            buffer = io.BytesIO()
            result.save(buffer, format='JPEG', quality=90)
            result_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # 响应
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            response = {
                "success": True,
                "image": result_b64,
                "params": {
                    "iso": iso,
                    "mode": mode,
                    "intensity": intensity or "auto"
                }
            }
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())

    def do_OPTIONS(self):
        """CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
