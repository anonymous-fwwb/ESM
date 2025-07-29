import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import subprocess
import json
import threading
import tempfile
import shutil
from pathlib import Path
import time
import torch

try:
    from torchvision.io import read_video
    from vmaf_torch import VMAF

    VMAF_AVAILABLE = True
except ImportError:
    VMAF_AVAILABLE = False


class DASHVMAFAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("DASH VMAF分析工具")
        self.root.geometry("800x700")

        # 码率配置 - 6个码率级别
        self.bitrate_configs = [
            {"name": "0.3Mbps", "bitrate": "300k", "maxrate": "450k", "bufsize": "600k"},
            {"name": "0.75Mbps", "bitrate": "750k", "maxrate": "1125k", "bufsize": "1500k"},
            {"name": "1.2Mbps", "bitrate": "1200k", "maxrate": "1800k", "bufsize": "2400k"},
            {"name": "1.8Mbps", "bitrate": "1800k", "maxrate": "2700k", "bufsize": "3600k"},
            {"name": "2.8Mbps", "bitrate": "2800k", "maxrate": "4200k", "bufsize": "5600k"},
            {"name": "4.3Mbps", "bitrate": "4300k", "maxrate": "6450k", "bufsize": "8600k"}
        ]

        self.segment_duration = 4  # 秒
        self.input_file = ""
        self.output_dir = ""
        self.vmaf_model = None

        # 检查依赖
        self.check_dependencies()

        # 创建界面
        self.create_widgets()

    def check_dependencies(self):
        """检查必要的依赖是否安装"""
        # 检查ffmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            self.ffmpeg_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.ffmpeg_available = False

        # 检查CUDA
        self.cuda_available = torch.cuda.is_available()

    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 输入文件选择
        ttk.Label(main_frame, text="输入MP4文件:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.input_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="浏览", command=self.select_input_file).grid(row=0, column=2)

        # 输出目录选择
        ttk.Label(main_frame, text="输出目录:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.output_var, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="浏览", command=self.select_output_dir).grid(row=1, column=2)

        # 分段时长设置
        ttk.Label(main_frame, text="分段时长(秒):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.segment_var = tk.StringVar(value="4")
        ttk.Entry(main_frame, textvariable=self.segment_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5)

        # 视频分辨率设置
        ttk.Label(main_frame, text="输出分辨率:").grid(row=3, column=0, sticky=tk.W, pady=5)
        res_frame = ttk.Frame(main_frame)
        res_frame.grid(row=3, column=1, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(res_frame, text="宽度:").grid(row=0, column=0, padx=5)
        self.width_var = tk.StringVar(value="1920")
        ttk.Entry(res_frame, textvariable=self.width_var, width=8).grid(row=0, column=1, padx=5)

        ttk.Label(res_frame, text="高度:").grid(row=0, column=2, padx=5)
        self.height_var = tk.StringVar(value="1080")
        ttk.Entry(res_frame, textvariable=self.height_var, width=8).grid(row=0, column=3, padx=5)

        ttk.Button(res_frame, text="自动检测", command=self.auto_detect_resolution).grid(row=0, column=4, padx=5)

        # 码率配置选择
        ttk.Label(main_frame, text="编码码率:").grid(row=4, column=0, sticky=tk.W, pady=5)
        bitrate_frame = ttk.Frame(main_frame)
        bitrate_frame.grid(row=4, column=1, columnspan=2, sticky=tk.W, pady=5)

        self.bitrate_vars = []
        for i, config in enumerate(self.bitrate_configs):
            var = tk.BooleanVar(value=True)
            self.bitrate_vars.append(var)
            ttk.Checkbutton(bitrate_frame, text=config['name'],
                            variable=var).grid(row=i // 3, column=i % 3, padx=10, sticky=tk.W)

        # 处理按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=20)

        self.process_btn = ttk.Button(button_frame, text="开始处理", command=self.start_processing)
        self.process_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(button_frame, text="停止", command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # 进度条
        ttk.Label(main_frame, text="处理进度:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # 状态显示
        ttk.Label(main_frame, text="状态信息:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.status_text = scrolledtext.ScrolledText(main_frame, height=15, width=80)
        self.status_text.grid(row=8, column=0, columnspan=3, pady=5)

        # 依赖状态显示
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=9, column=0, columnspan=3, pady=10)

        ffmpeg_status = "✓" if self.ffmpeg_available else "✗"
        vmaf_status = "✓" if VMAF_AVAILABLE else "✗"
        cuda_status = "✓" if self.cuda_available else "✗"

        ttk.Label(status_frame,
                  text=f"FFmpeg: {ffmpeg_status} | VMAF-torch: {vmaf_status} | CUDA: {cuda_status}").pack()

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(8, weight=1)

        self.processing = False

    def select_input_file(self):
        """选择输入文件"""
        filename = filedialog.askopenfilename(
            title="选择MP4文件",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if filename:
            self.input_var.set(filename)
            self.input_file = filename
            # 自动检测分辨率
            self.auto_detect_resolution()

    def select_output_dir(self):
        """选择输出目录"""
        dirname = filedialog.askdirectory(title="选择输出目录")
        if dirname:
            self.output_var.set(dirname)
            self.output_dir = dirname

    def auto_detect_resolution(self):
        """自动检测视频分辨率"""
        if not self.input_file or not os.path.exists(self.input_file):
            return

        try:
            video_info = self.get_video_info(self.input_file)
            for stream in video_info['streams']:
                if stream['codec_type'] == 'video':
                    width = stream.get('width', 1920)
                    height = stream.get('height', 1080)
                    self.width_var.set(str(width))
                    self.height_var.set(str(height))
                    self.log_message(f"检测到视频分辨率: {width}x{height}")
                    break
        except Exception as e:
            self.log_message(f"自动检测分辨率失败: {str(e)}")

    def log_message(self, message):
        """添加日志消息"""
        self.status_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update()

    def start_processing(self):
        """开始处理"""
        if not self.validate_inputs():
            return

        self.processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress.start()

        # 在新线程中处理，避免界面冻结
        threading.Thread(target=self.process_video, daemon=True).start()

    def stop_processing(self):
        """停止处理"""
        self.processing = False
        self.process_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress.stop()
        self.log_message("处理已停止")

    def validate_inputs(self):
        """验证输入"""
        if not self.ffmpeg_available:
            messagebox.showerror("错误", "FFmpeg未安装或不在PATH中")
            return False

        if not VMAF_AVAILABLE:
            messagebox.showerror("错误", "VMAF-torch或torchvision未安装")
            return False

        if not self.input_file or not os.path.exists(self.input_file):
            messagebox.showerror("错误", "请选择有效的输入文件")
            return False

        if not self.output_dir:
            messagebox.showerror("错误", "请选择输出目录")
            return False

        if not any(var.get() for var in self.bitrate_vars):
            messagebox.showerror("错误", "请至少选择一个码率")
            return False

        try:
            self.segment_duration = int(self.segment_var.get())
            if self.segment_duration <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("错误", "分段时长必须是正整数")
            return False

        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            if width <= 0 or height <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("错误", "分辨率必须是正整数")
            return False

        return True

    def get_video_info(self, video_path):
        """获取视频信息"""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)

    def encode_video_bitrate(self, input_file, output_file, width, height, bitrate_config):
        """编码指定码率的视频"""
        cmd = [
            "ffmpeg", "-y", "-i", input_file,
            "-c:v", "libx264",
            "-b:v", bitrate_config["bitrate"],
            "-maxrate", bitrate_config["maxrate"],
            "-bufsize", bitrate_config["bufsize"],
            "-vf", f"scale={width}:{height}",
            "-c:a", "aac", "-b:a", "128k",
            "-preset", "medium", "-crf", "23",
            "-x264-params", "keyint=48:min-keyint=48:scenecut=0",  # 固定GOP
            output_file
        ]

        subprocess.run(cmd, check=True, capture_output=True)

    def segment_video(self, input_file, output_pattern, segment_duration):
        """分割视频为固定时长的片段"""
        cmd = [
            "ffmpeg", "-y", "-i", input_file,
            "-c", "copy", "-f", "segment",
            "-segment_time", str(segment_duration),
            "-reset_timestamps", "1",
            output_pattern
        ]

        subprocess.run(cmd, check=True, capture_output=True)

    def load_mp4_as_Y(self, path, max_frames=None):
        """加载MP4文件并转换为Y通道"""
        try:
            # read_video 返回 video(T,H,W,3), audio, info
            video, _, _ = read_video(path, pts_unit="sec")
            if max_frames:
                video = video[:max_frames]
            # video 是 uint8, 转 float 并归一到 [0,255]
            video = video.permute(0, 3, 1, 2).float()  # [T,3,H,W]
            # RGB -> Y (BT.601)：Y = 0.299 R + 0.587 G + 0.114 B
            R, G, B = video.unbind(1)
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            return Y.unsqueeze(1)  # [T,1,H,W]
        except Exception as e:
            self.log_message(f"加载视频失败 {path}: {str(e)}")
            return None

    def calculate_vmaf_segments(self, ref_segments_dir, dist_segments_dir, segment_files):
        """计算片段的VMAF分数"""
        if self.vmaf_model is None:
            self.vmaf_model = VMAF()
            if self.cuda_available:
                self.vmaf_model = self.vmaf_model.to("cuda")

        device = "cuda" if self.cuda_available else "cpu"
        segment_results = []

        for i, segment_file in enumerate(segment_files):
            if not self.processing:
                break

            ref_path = os.path.join(ref_segments_dir, segment_file)
            dist_path = os.path.join(dist_segments_dir, segment_file)

            if not (os.path.exists(ref_path) and os.path.exists(dist_path)):
                self.log_message(f"警告: 片段文件缺失 {segment_file}")
                continue

            try:
                # 加载参考和失真视频的Y通道
                ref_Y = self.load_mp4_as_Y(ref_path)
                dist_Y = self.load_mp4_as_Y(dist_path)

                if ref_Y is None or dist_Y is None:
                    continue

                # 确保帧数一致
                min_frames = min(ref_Y.shape[0], dist_Y.shape[0])
                ref_Y = ref_Y[:min_frames].to(device)
                dist_Y = dist_Y[:min_frames].to(device)

                # 计算VMAF分数
                scores = self.vmaf_model(ref_Y, dist_Y)
                avg_score = scores.mean().item()

                segment_result = {
                    'segment': segment_file,
                    'segment_index': i,
                    'start_time': i * self.segment_duration,
                    'duration': self.segment_duration,
                    'vmaf_score': avg_score,
                    'frames_count': min_frames
                }
                segment_results.append(segment_result)

                self.log_message(f"片段 {i + 1}/{len(segment_files)}: {segment_file} VMAF = {avg_score:.2f}")

            except Exception as e:
                self.log_message(f"计算VMAF失败 {segment_file}: {str(e)}")
                continue

        return segment_results

    def process_video(self):
        """主处理流程"""
        try:
            self.log_message("开始处理视频...")

            # 创建输出目录结构
            project_dir = os.path.join(self.output_dir, "dash_output")
            os.makedirs(project_dir, exist_ok=True)

            # 获取输出分辨率
            width = int(self.width_var.get())
            height = int(self.height_var.get())

            # 获取选中的码率配置
            selected_configs = [config for i, config in enumerate(self.bitrate_configs) if self.bitrate_vars[i].get()]

            if not selected_configs:
                self.log_message("错误: 没有选择任何码率配置")
                return

            self.log_message(f"输出分辨率: {width}x{height}")
            self.log_message(f"选择的码率: {[c['name'] for c in selected_configs]}")

            # 第一步：编码所有码率版本
            encoded_files = {}
            for config in selected_configs:
                if not self.processing:
                    break

                self.log_message(f"编码 {config['name']}...")
                bitrate_dir = os.path.join(project_dir, config['name'])
                os.makedirs(bitrate_dir, exist_ok=True)

                encoded_file = os.path.join(bitrate_dir, f"encoded_{config['name']}.mp4")
                self.encode_video_bitrate(self.input_file, encoded_file, width, height, config)
                encoded_files[config['name']] = encoded_file

            # 第二步：分割所有版本
            segment_dirs = {}
            for config in selected_configs:
                if not self.processing:
                    break

                config_name = config['name']
                self.log_message(f"分割 {config_name}...")

                bitrate_dir = os.path.join(project_dir, config_name)
                segment_dir = os.path.join(bitrate_dir, "segments")
                os.makedirs(segment_dir, exist_ok=True)

                segment_pattern = os.path.join(segment_dir, "segment_%03d.mp4")
                self.segment_video(encoded_files[config_name], segment_pattern, self.segment_duration)
                segment_dirs[config_name] = segment_dir

            # 获取分片文件列表（使用最高码率的分片作为基准）
            reference_config = selected_configs[-1]  # 最高码率作为参考
            reference_segments_dir = segment_dirs[reference_config['name']]
            segment_files = sorted([f for f in os.listdir(reference_segments_dir) if f.endswith('.mp4')])

            self.log_message(f"使用 {reference_config['name']} 作为VMAF参考，共 {len(segment_files)} 个片段")

            # 第三步：计算VMAF
            results = []
            for config in selected_configs:
                if not self.processing:
                    break

                config_name = config['name']

                if config_name == reference_config['name']:
                    # 参考版本，VMAF分数为100（理论上）
                    self.log_message(f"{config_name}: 作为参考版本，跳过VMAF计算")
                    segment_results = []
                    for i, segment_file in enumerate(segment_files):
                        segment_results.append({
                            'segment': segment_file,
                            'segment_index': i,
                            'start_time': i * self.segment_duration,
                            'duration': self.segment_duration,
                            'vmaf_score': 100.0,  # 参考版本
                            'frames_count': 0
                        })
                else:
                    # 计算与参考版本的VMAF
                    self.log_message(f"计算 {config_name} 的VMAF分数...")
                    segment_results = self.calculate_vmaf_segments(
                        reference_segments_dir,
                        segment_dirs[config_name],
                        segment_files
                    )

                # 计算平均VMAF
                valid_scores = [s['vmaf_score'] for s in segment_results if s['vmaf_score'] is not None]
                avg_vmaf = sum(valid_scores) / len(valid_scores) if valid_scores else 0

                config_result = {
                    'bitrate_config': config_name,
                    'bitrate': config['bitrate'],
                    'width': width,
                    'height': height,
                    'segments': segment_results,
                    'avg_vmaf': avg_vmaf,
                    'is_reference': (config_name == reference_config['name'])
                }
                results.append(config_result)

                # 保存单个配置的结果
                bitrate_dir = os.path.join(project_dir, config_name)
                with open(os.path.join(bitrate_dir, 'vmaf_results.json'), 'w', encoding='utf-8') as f:
                    json.dump(config_result, f, indent=2, ensure_ascii=False)

            # 保存完整结果
            final_results = {
                'input_file': self.input_file,
                'output_resolution': f"{width}x{height}",
                'segment_duration': self.segment_duration,
                'reference_config': reference_config['name'],
                'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'bitrate_configs': results
            }

            with open(os.path.join(project_dir, 'complete_results.json'), 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)

            self.log_message("处理完成！")
            self.log_message(f"结果保存在: {project_dir}")

            # 显示摘要
            self.log_message("\n=== 处理摘要 ===")
            for result in results:
                if result['is_reference']:
                    self.log_message(f"{result['bitrate_config']}: 参考版本")
                else:
                    self.log_message(f"{result['bitrate_config']}: 平均VMAF = {result['avg_vmaf']:.2f}")

        except Exception as e:
            self.log_message(f"处理错误: {str(e)}")
            messagebox.showerror("错误", f"处理过程中发生错误: {str(e)}")

        finally:
            self.processing = False
            self.process_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.progress.stop()


if __name__ == "__main__":
    root = tk.Tk()
    app = DASHVMAFAnalyzer(root)
    root.mainloop()