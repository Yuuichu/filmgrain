#!/usr/bin/env python3
"""
Film Grain Effect Generator - GUI Version
胶片颗粒效果生成器图形界面
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
from threading import Thread
import time

from filmgrain import FilmGrain


class FilmGrainGUI:
    """胶片颗粒效果 GUI"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Film Grain Generator - 胶片颗粒生成器")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)

        # 状态变量
        self.original_image: Image.Image = None
        self.processed_image: Image.Image = None
        self.current_file: str = None
        self.preview_job = None

        # 参数变量
        self.iso_var = tk.IntVar(value=400)
        self.intensity_var = tk.DoubleVar(value=0.0)  # 0 = 使用ISO预设
        self.color_mode_var = tk.StringVar(value="color")
        self.auto_preview_var = tk.BooleanVar(value=True)
        self.seed_var = tk.StringVar(value="")

        self._setup_ui()
        self._bind_events()

    def _setup_ui(self):
        """构建界面"""
        # 主容器
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧: 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="参数设置", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self._setup_controls(control_frame)

        # 右侧: 图像预览
        preview_frame = ttk.LabelFrame(main_frame, text="预览", padding=5)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._setup_preview(preview_frame)

    def _setup_controls(self, parent):
        """设置控制面板"""
        # 文件操作
        file_frame = ttk.LabelFrame(parent, text="文件", padding=5)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="打开图像...", command=self._open_image).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="保存结果...", command=self._save_image).pack(fill=tk.X, pady=2)

        # ISO 设置
        iso_frame = ttk.LabelFrame(parent, text="ISO 感光度", padding=5)
        iso_frame.pack(fill=tk.X, pady=(0, 10))

        self.iso_label = ttk.Label(iso_frame, text="ISO 400", font=("", 12, "bold"))
        self.iso_label.pack()

        iso_scale = ttk.Scale(
            iso_frame,
            from_=0,
            to=6,
            orient=tk.HORIZONTAL,
            command=self._on_iso_change
        )
        iso_scale.set(3)  # 默认 400
        iso_scale.pack(fill=tk.X, pady=5)

        # ISO 刻度标签
        iso_ticks = ttk.Frame(iso_frame)
        iso_ticks.pack(fill=tk.X)
        for i, iso in enumerate([50, 100, 200, 400, 800, 1600, 3200]):
            lbl = ttk.Label(iso_ticks, text=str(iso), font=("", 8))
            lbl.place(relx=i/6, anchor=tk.N)

        # 颗粒强度
        intensity_frame = ttk.LabelFrame(parent, text="颗粒强度", padding=5)
        intensity_frame.pack(fill=tk.X, pady=(0, 10))

        self.intensity_label = ttk.Label(intensity_frame, text="自动 (ISO 预设)")
        self.intensity_label.pack()

        self.intensity_scale = ttk.Scale(
            intensity_frame,
            from_=0,
            to=0.3,
            orient=tk.HORIZONTAL,
            command=self._on_intensity_change
        )
        self.intensity_scale.set(0)
        self.intensity_scale.pack(fill=tk.X, pady=5)

        ttk.Button(
            intensity_frame,
            text="重置为自动",
            command=lambda: self.intensity_scale.set(0)
        ).pack()

        # 胶片类型
        mode_frame = ttk.LabelFrame(parent, text="胶片类型", padding=5)
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(
            mode_frame,
            text="🎨 彩色胶片 (染料云)",
            variable=self.color_mode_var,
            value="color",
            command=self._on_param_change
        ).pack(anchor=tk.W)

        ttk.Radiobutton(
            mode_frame,
            text="⬛ 黑白胶片 (银盐晶体)",
            variable=self.color_mode_var,
            value="bw",
            command=self._on_param_change
        ).pack(anchor=tk.W)

        # 高级选项
        adv_frame = ttk.LabelFrame(parent, text="高级选项", padding=5)
        adv_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(
            adv_frame,
            text="实时预览",
            variable=self.auto_preview_var
        ).pack(anchor=tk.W)

        seed_row = ttk.Frame(adv_frame)
        seed_row.pack(fill=tk.X, pady=5)
        ttk.Label(seed_row, text="随机种子:").pack(side=tk.LEFT)
        ttk.Entry(seed_row, textvariable=self.seed_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(seed_row, text="随机", command=self._randomize_seed, width=6).pack(side=tk.LEFT)

        # 操作按钮
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            action_frame,
            text="🔄 应用效果",
            command=self._apply_effect
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            action_frame,
            text="↩️ 重置原图",
            command=self._reset_preview
        ).pack(fill=tk.X, pady=2)

        # 状态栏
        self.status_label = ttk.Label(parent, text="请打开一张图像", foreground="gray")
        self.status_label.pack(side=tk.BOTTOM, pady=10)

    def _setup_preview(self, parent):
        """设置预览区域"""
        # 预览画布
        self.canvas = tk.Canvas(parent, bg="#2a2a2a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 绑定画布大小变化
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # 对比模式标签
        self.compare_label = ttk.Label(
            parent,
            text="提示: 按住空格键查看原图对比",
            foreground="gray"
        )
        self.compare_label.pack(side=tk.BOTTOM)

    def _bind_events(self):
        """绑定事件"""
        self.root.bind("<space>", self._show_original)
        self.root.bind("<KeyRelease-space>", self._show_processed)
        self.root.bind("<Control-o>", lambda e: self._open_image())
        self.root.bind("<Control-s>", lambda e: self._save_image())

    def _open_image(self):
        """打开图像文件"""
        filetypes = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("所有文件", "*.*")
        ]
        filepath = filedialog.askopenfilename(filetypes=filetypes)

        if filepath:
            try:
                self.original_image = Image.open(filepath).convert("RGB")
                self.processed_image = None
                self.current_file = filepath

                self._update_preview(self.original_image)
                self._update_status(f"已加载: {Path(filepath).name}")

                # 自动应用效果
                if self.auto_preview_var.get():
                    self._apply_effect()

            except Exception as e:
                messagebox.showerror("错误", f"无法打开图像:\n{e}")

    def _save_image(self):
        """保存处理后的图像"""
        if self.processed_image is None:
            messagebox.showwarning("提示", "请先应用效果")
            return

        # 默认文件名
        if self.current_file:
            default_name = Path(self.current_file).stem + "_grain.jpg"
        else:
            default_name = "grain_output.jpg"

        filetypes = [
            ("JPEG", "*.jpg"),
            ("PNG", "*.png"),
            ("所有文件", "*.*")
        ]
        filepath = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=filetypes
        )

        if filepath:
            try:
                self.processed_image.save(filepath, quality=95)
                self._update_status(f"已保存: {Path(filepath).name}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败:\n{e}")

    def _apply_effect(self):
        """应用胶片颗粒效果"""
        if self.original_image is None:
            messagebox.showwarning("提示", "请先打开一张图像")
            return

        self._update_status("处理中...")

        def process():
            try:
                # 获取参数
                iso = self.iso_var.get()
                intensity = self.intensity_var.get()
                color_mode = self.color_mode_var.get()

                # 解析种子
                seed = None
                if self.seed_var.get().strip():
                    try:
                        seed = int(self.seed_var.get())
                    except ValueError:
                        pass

                # 创建颗粒生成器
                grain = FilmGrain(iso=iso, color_mode=color_mode, seed=seed)

                # 应用效果
                intensity_override = intensity if intensity > 0 else None
                self.processed_image = grain.apply(self.original_image, intensity_override)

                # 更新预览
                self.root.after(0, lambda: self._update_preview(self.processed_image))
                self.root.after(0, lambda: self._update_status(
                    f"ISO {iso} | {'黑白' if color_mode == 'bw' else '彩色'} | "
                    f"强度 {intensity:.2f}" if intensity > 0 else f"ISO {iso} | {'黑白' if color_mode == 'bw' else '彩色'}"
                ))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"处理失败:\n{e}"))
                self.root.after(0, lambda: self._update_status("处理失败"))

        # 在后台线程处理
        Thread(target=process, daemon=True).start()

    def _update_preview(self, image: Image.Image):
        """更新预览图像"""
        if image is None:
            return

        # 获取画布尺寸
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        # 计算缩放比例 (保持比例)
        img_width, img_height = image.size
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        # 缩放图像
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(resized)

        # 更新画布
        self.canvas.delete("all")
        x = canvas_width // 2
        y = canvas_height // 2
        self.canvas.create_image(x, y, image=self.preview_photo, anchor=tk.CENTER)

    def _reset_preview(self):
        """重置为原图"""
        if self.original_image:
            self.processed_image = None
            self._update_preview(self.original_image)
            self._update_status("已重置为原图")

    def _show_original(self, event=None):
        """显示原图 (按住空格)"""
        if self.original_image and self.processed_image:
            self._update_preview(self.original_image)

    def _show_processed(self, event=None):
        """显示处理后图像"""
        if self.processed_image:
            self._update_preview(self.processed_image)

    def _on_canvas_resize(self, event):
        """画布大小变化时更新预览"""
        if self.processed_image:
            self._update_preview(self.processed_image)
        elif self.original_image:
            self._update_preview(self.original_image)

    def _on_iso_change(self, value):
        """ISO 滑块变化"""
        iso_values = [50, 100, 200, 400, 800, 1600, 3200]
        index = int(float(value))
        iso = iso_values[index]
        self.iso_var.set(iso)
        self.iso_label.config(text=f"ISO {iso}")
        self._on_param_change()

    def _on_intensity_change(self, value):
        """强度滑块变化"""
        intensity = float(value)
        self.intensity_var.set(intensity)

        if intensity == 0:
            self.intensity_label.config(text="自动 (ISO 预设)")
        else:
            self.intensity_label.config(text=f"手动: {intensity:.2%}")

        self._on_param_change()

    def _on_param_change(self):
        """参数变化时触发"""
        if self.auto_preview_var.get() and self.original_image:
            # 延迟执行，避免频繁更新
            if self.preview_job:
                self.root.after_cancel(self.preview_job)
            self.preview_job = self.root.after(200, self._apply_effect)

    def _randomize_seed(self):
        """生成随机种子"""
        import random
        self.seed_var.set(str(random.randint(1, 999999)))
        self._on_param_change()

    def _update_status(self, text: str):
        """更新状态栏"""
        self.status_label.config(text=text)


def main():
    root = tk.Tk()

    # 设置样式
    style = ttk.Style()
    style.theme_use("clam")  # 使用现代主题

    app = FilmGrainGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
