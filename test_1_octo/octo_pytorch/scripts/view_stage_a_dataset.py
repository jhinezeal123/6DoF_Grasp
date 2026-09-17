#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageTk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from octo.data.stage_a_statistics import summarize_raw_episode


class StageAViewerApp:
    def __init__(self, root: tk.Tk, raw_dir: Path, statistics_json: Path | None = None):
        self.root = root
        self.root.title("Stage-A Dataset Viewer v0.0.4")
        self.raw_dir = raw_dir
        self.statistics_json = statistics_json
        self.episode_dirs = sorted(p for p in self.raw_dir.glob("episode_*") if p.is_dir())
        if not self.episode_dirs:
            raise FileNotFoundError(f"No episode_* directories found under {self.raw_dir}")

        self.dataset_stats = self._load_stats(statistics_json)
        self.episode_index = 0
        self.step_index = 0
        self.current_episode = None
        self.current_payload = None
        self.current_image_ref = None

        self._build_ui()
        self._load_episode(0)

    def _load_stats(self, statistics_json: Path | None):
        if statistics_json is None:
            candidate = self.raw_dir.parent / "tfds" / "myarm_stage_a_dataset_statistics.json"
            if candidate.exists():
                statistics_json = candidate
            else:
                candidate = self.raw_dir.parent / "myarm_stage_a_dataset_statistics.json"
                if candidate.exists():
                    statistics_json = candidate
        if statistics_json is None or not statistics_json.exists():
            return None
        with open(statistics_json, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_ui(self):
        self.root.geometry("1280x860")
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Raw dir:").grid(row=0, column=0, sticky="w")
        self.raw_dir_var = tk.StringVar(value=str(self.raw_dir))
        ttk.Entry(top, textvariable=self.raw_dir_var, width=90).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="Open", command=self._choose_raw_dir).grid(row=0, column=2, padx=4)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Episode:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.episode_var = tk.StringVar()
        self.episode_combo = ttk.Combobox(top, textvariable=self.episode_var, state="readonly", width=40)
        self.episode_combo["values"] = [ep.name for ep in self.episode_dirs]
        self.episode_combo.grid(row=1, column=1, sticky="w", padx=4, pady=(6, 0))
        self.episode_combo.bind("<<ComboboxSelected>>", self._on_episode_selected)

        control = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        control.pack(fill=tk.X)
        ttk.Button(control, text="<< Prev Episode", command=lambda: self._load_episode(self.episode_index - 1)).pack(side=tk.LEFT)
        ttk.Button(control, text="Next Episode >>", command=lambda: self._load_episode(self.episode_index + 1)).pack(side=tk.LEFT, padx=6)
        ttk.Separator(control, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(control, text="< Prev Step", command=lambda: self._set_step(self.step_index - 1)).pack(side=tk.LEFT)
        ttk.Button(control, text="Next Step >", command=lambda: self._set_step(self.step_index + 1)).pack(side=tk.LEFT, padx=6)

        self.step_slider = tk.Scale(control, from_=0, to=0, orient=tk.HORIZONTAL, length=420, command=self._on_slider)
        self.step_slider.pack(side=tk.LEFT, padx=8)
        self.step_label = ttk.Label(control, text="step 0 / 0")
        self.step_label.pack(side=tk.LEFT)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        self.image_label = ttk.Label(left)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        right_top = ttk.LabelFrame(right, text="Step / Episode Details", padding=8)
        right_top.pack(fill=tk.BOTH, expand=True)
        self.details_text = tk.Text(right_top, wrap="word", width=52)
        self.details_text.pack(fill=tk.BOTH, expand=True)

        right_bottom = ttk.LabelFrame(right, text="Dataset Statistics", padding=8)
        right_bottom.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.stats_text = tk.Text(right_bottom, wrap="word", height=18)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self._refresh_dataset_stats_panel()

    def _refresh_dataset_stats_panel(self):
        self.stats_text.configure(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        if not self.dataset_stats:
            self.stats_text.insert(tk.END, "No dataset statistics JSON found.\n\nRun prepare_stage_a_dataset.py to export statistics and summary files.")
        else:
            lines = []
            lines.append(f"dataset_name: {self.dataset_stats.get('dataset_name', '')}")
            lines.append(f"num_trajectories: {self.dataset_stats.get('num_trajectories', '')}")
            lines.append(f"num_transitions: {self.dataset_stats.get('num_transitions', '')}")
            lines.append("")
            for key in ("action", "proprio"):
                if key not in self.dataset_stats:
                    continue
                lines.append(f"[{key}]")
                stats = self.dataset_stats[key]
                for stat_name in ("mean", "std", "min", "max", "p01", "p99", "mask"):
                    if stat_name in stats:
                        lines.append(f"  {stat_name}: {np.array(stats[stat_name])}")
                lines.append("")
            self.stats_text.insert(tk.END, "\n".join(lines).strip())
        self.stats_text.configure(state=tk.DISABLED)

    def _choose_raw_dir(self):
        path = filedialog.askdirectory(initialdir=str(self.raw_dir))
        if not path:
            return
        self.raw_dir = Path(path)
        self.raw_dir_var.set(str(self.raw_dir))
        self.episode_dirs = sorted(p for p in self.raw_dir.glob("episode_*") if p.is_dir())
        self.episode_combo["values"] = [ep.name for ep in self.episode_dirs]
        if not self.episode_dirs:
            self.details_text.configure(state=tk.NORMAL)
            self.details_text.delete("1.0", tk.END)
            self.details_text.insert(tk.END, f"No episode_* directories found under {self.raw_dir}\n")
            self.details_text.configure(state=tk.DISABLED)
            return
        self._load_episode(0)

    def _on_episode_selected(self, _event=None):
        name = self.episode_var.get()
        for idx, episode_dir in enumerate(self.episode_dirs):
            if episode_dir.name == name:
                self._load_episode(idx)
                break

    def _on_slider(self, value):
        try:
            idx = int(float(value))
        except Exception:
            return
        if self.current_payload is None:
            return
        if idx != self.step_index:
            self._set_step(idx)

    def _load_episode(self, episode_index: int):
        if not self.episode_dirs:
            return
        episode_index = max(0, min(episode_index, len(self.episode_dirs) - 1))
        self.episode_index = episode_index
        episode_dir = self.episode_dirs[self.episode_index]
        self.current_episode = episode_dir

        summary = summarize_raw_episode(episode_dir)
        achieved = np.load(episode_dir / "achieved_state.npy").astype(np.float32)
        commanded = np.load(episode_dir / "commanded_state.npy").astype(np.float32)
        action_path = episode_dir / "delta_action_from_get_coords.npy"
        if not action_path.exists():
            action_path = episode_dir / "action.npy"
        action = np.load(action_path).astype(np.float32)
        timestamps = np.load(episode_dir / "timestamps.npy").astype(np.float32)
        frame_paths = sorted((episode_dir / "frames" / "image_primary").glob("*.jpg"))
        meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
        instruction = (episode_dir / "instruction.txt").read_text(encoding="utf-8").strip()

        usable_steps = min(len(achieved), len(commanded), len(timestamps), len(frame_paths))
        usable_actions = min(max(usable_steps - 1, 0), len(action))
        self.current_payload = {
            "summary": summary,
            "meta": meta,
            "instruction": instruction,
            "achieved": achieved[:usable_steps],
            "commanded": commanded[:usable_steps],
            "action": action[:usable_actions],
            "timestamps": timestamps[:usable_steps],
            "frame_paths": frame_paths[:usable_steps],
        }

        self.episode_var.set(episode_dir.name)
        self.step_slider.configure(to=max(usable_steps - 1, 0))
        self._set_step(0)

    def _set_step(self, step_index: int):
        if self.current_payload is None:
            return
        usable_steps = len(self.current_payload["achieved"])
        if usable_steps <= 0:
            return
        self.step_index = max(0, min(step_index, usable_steps - 1))
        self.step_slider.set(self.step_index)
        self.step_label.configure(text=f"step {self.step_index} / {usable_steps - 1}")
        self._render_image()
        self._render_details()

    def _render_image(self):
        frame_path = self.current_payload["frame_paths"][self.step_index]
        image = Image.open(frame_path).convert("RGB")
        image.thumbnail((760, 760))
        self.current_image_ref = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.current_image_ref)

    def _render_details(self):
        payload = self.current_payload
        achieved = payload["achieved"][self.step_index]
        commanded = payload["commanded"][self.step_index]
        tracking_error = achieved - commanded
        timestamp_s = float(payload["timestamps"][self.step_index])
        action = payload["action"][self.step_index] if self.step_index < len(payload["action"]) else None
        summary = payload["summary"]
        meta = payload["meta"]

        lines = []
        lines.append(f"episode: {self.current_episode.name}")
        lines.append(f"instruction: {payload['instruction']}")
        lines.append(f"timestamp_s: {timestamp_s:.4f}")
        lines.append(f"control_hz(meta): {meta.get('control_hz', '')}")
        lines.append("")
        lines.append("achieved_state_7d [x,y,z,rx,ry,rz,grip]:")
        lines.append(f"  {np.array2string(achieved, precision=6)}")
        lines.append("commanded_state_7d [x,y,z,rx,ry,rz,grip]:")
        lines.append(f"  {np.array2string(commanded, precision=6)}")
        lines.append("tracking_error = achieved - commanded:")
        lines.append(f"  {np.array2string(tracking_error, precision=6)}")
        if action is not None:
            lines.append("delta_action_from_get_coords[step]:")
            lines.append(f"  {np.array2string(action, precision=6)}")
        else:
            lines.append("delta_action_from_get_coords[step]: <N/A on final step>")
        if self.step_index > 0:
            prev_achieved = payload["achieved"][self.step_index - 1]
            lines.append("delta achieved from previous step:")
            lines.append(f"  {np.array2string(achieved - prev_achieved, precision=6)}")
        lines.append("")
        lines.append("episode summary:")
        lines.append(f"  num_steps={summary.num_steps}, num_frames={summary.num_frames}, duration_s={summary.duration_s:.4f}, approx_control_hz={summary.approx_control_hz:.4f}")
        lines.append(f"  action_mean={np.array(summary.action_mean)}")
        lines.append(f"  action_std ={np.array(summary.action_std)}")
        lines.append(f"  tracking_error_mean={np.array(summary.commanded_tracking_error_mean)}")
        lines.append(f"  tracking_error_max_abs={np.array(summary.commanded_tracking_error_max_abs)}")

        self.details_text.configure(state=tk.NORMAL)
        self.details_text.delete("1.0", tk.END)
        self.details_text.insert(tk.END, "\n".join(lines))
        self.details_text.configure(state=tk.DISABLED)


def _parse_args():
    ap = argparse.ArgumentParser(description="Simple Tkinter viewer for Stage-A raw dataset")
    ap.add_argument("--raw_dir", default=None, help="Path to raw directory containing episode_* folders")
    ap.add_argument("--statistics_json", default=None, help="Optional path to myarm_stage_a_dataset_statistics.json")
    return ap.parse_args()


def main():
    args = _parse_args()
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else Path.cwd()
    root = tk.Tk()
    app = StageAViewerApp(
        root=root,
        raw_dir=raw_dir,
        statistics_json=Path(args.statistics_json).resolve() if args.statistics_json else None,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
