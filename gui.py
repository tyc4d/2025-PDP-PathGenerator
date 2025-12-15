"""
GUI (Tkinter) for interactive profile tuning + preview (mm units) and export to DXF/SVG.

Run:
  python gui.py
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import main as gen


Point = Tuple[float, float]  # (x_mm, y_mm) after scaling rules


@dataclass
class RawCacheKey:
    input_path: str
    input_format: str
    sample_step_m: float
    elevation_source: str


class ProfileGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("中央山脈主稜線剖面產生器（mm）")
        self.geometry("1150x720")

        # internal state
        self.input_path: Optional[Path] = None
        self.input_format: str = "auto"
        self._raw_cache_key: Optional[RawCacheKey] = None
        self._raw_samples: Optional[List[gen.Sample]] = None
        self._compute_thread: Optional[threading.Thread] = None
        self._busy = False

        # default: keep logs quiet
        gen.configure_logging(verbose=False)

        # vars
        self.var_input = tk.StringVar(value="")
        self.var_status = tk.StringVar(value="尚未載入資料")

        # Use StringVar for numeric inputs so users can type partial values like "0." without Tk errors.
        self.var_sample_step_m = tk.StringVar(value="200")
        self.var_smooth_window_m = tk.StringVar(value="2000")
        self.var_simplify_eps_mm = tk.StringVar(value="1.0")
        self.var_vert_scale = tk.StringVar(value="1.0")
        self.var_fit_width_mm = tk.StringVar(value="240")
        self.var_fit_height_mm = tk.StringVar(value="80")
        self.var_base_elevation_m = tk.StringVar(value="")  # empty => auto min

        # profile options
        self.var_show_profile_normal = tk.BooleanVar(value=True)
        self.var_show_profile_compressed = tk.BooleanVar(value=False)
        self.var_compress_gamma = tk.StringVar(value="0.6")  # 0<gamma<=1, smaller => more compression

        self.var_show_grid = tk.BooleanVar(value=True)

        # segment/splitting options (Tab 2)
        self.var_split_n = tk.StringVar(value="6")
        self.var_split_buffer_mm = tk.StringVar(value="10")
        self.var_split_index = tk.StringVar(value="1")  # 1-based
        self.var_split_taper = tk.BooleanVar(value=True)  # taper ends to baseline for smooth cut

        # debounce
        self._pending_after_id: Optional[str] = None

        self._build_ui()
        self._bind_var_traces()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        # top bar
        top = ttk.Frame(root)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        top.columnconfigure(1, weight=1)

        ttk.Button(top, text="開啟 KML/CSV", command=self.on_open).grid(row=0, column=0, padx=(0, 8))
        ttk.Entry(top, textvariable=self.var_input, state="readonly").grid(row=0, column=1, sticky="ew")
        ttk.Button(top, text="重新抽樣海拔（慢）", command=self.on_recompute_raw).grid(row=0, column=2, padx=(8, 0))

        # left controls (tabs)
        self.nb = ttk.Notebook(root)
        self.nb.grid(row=1, column=0, sticky="nsw", padx=(0, 10))

        tab_single = ttk.Frame(self.nb, padding=10)
        tab_split = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab_single, text="單一剖面")
        self.nb.add(tab_split, text="分段匯出")

        # ---------------- Tab 1: single profile ----------------
        r = 0
        r = self._row_text_number(tab_single, r, "sample-step-m（m）", self.var_sample_step_m, hint="KML/經緯度加密步距；越小越細但更慢")
        r = self._row_text_number(tab_single, r, "smooth-window-m（m）", self.var_smooth_window_m, hint="平滑窗；1000~5000 建議")
        r = self._row_text_number(tab_single, r, "simplify-epsilon（mm）", self.var_simplify_eps_mm, hint="折線簡化；0.3~2.0 建議")
        r = self._row_text_number(tab_single, r, "vert-scale（倍）", self.var_vert_scale, hint="垂直誇張倍率")
        r = self._row_text_number(tab_single, r, "fit-width（mm）", self.var_fit_width_mm, hint="整體輸出寬度（mm）（例如 240）")
        r = self._row_text_number(tab_single, r, "fit-height（mm）", self.var_fit_height_mm, hint="整體輸出高度（mm）（例如 80）")

        ttk.Separator(tab_single).grid(row=r, column=0, columnspan=2, sticky="ew", pady=10)
        r += 1

        ttk.Label(tab_single, text="Profile 顯示/匯出").grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 6))
        r += 1
        ttk.Checkbutton(tab_single, text="Profile A：原始（Normal）", variable=self.var_show_profile_normal).grid(
            row=r, column=0, columnspan=2, sticky="w"
        )
        r += 1
        ttk.Checkbutton(tab_single, text="Profile B：高度壓縮（Compressed）", variable=self.var_show_profile_compressed).grid(
            row=r, column=0, columnspan=2, sticky="w"
        )
        r += 1
        r = self._row_text_number(tab_single, r, "壓縮 gamma（0~1）", self.var_compress_gamma, hint="越小越壓縮（高峰差距變小）；建議 0.4~0.8")

        ttk.Label(tab_single, text="base-elevation-m（m）").grid(row=r, column=0, sticky="w", pady=(10, 2))
        ent_base = ttk.Entry(tab_single, textvariable=self.var_base_elevation_m, width=12)
        ent_base.grid(row=r, column=1, sticky="w", pady=(10, 2))
        ttk.Label(tab_single, text="空白=自動用最低點當 0").grid(row=r + 1, column=0, columnspan=2, sticky="w", pady=(0, 8))
        r += 2

        ttk.Checkbutton(tab_single, text="顯示格線（預覽用）", variable=self.var_show_grid).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(6, 8)
        )
        r += 1

        ttk.Button(tab_single, text="匯出 DXF…", command=self.on_export_dxf).grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1
        ttk.Button(tab_single, text="匯出 SVG…", command=self.on_export_svg).grid(row=r, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        # ---------------- Tab 2: split/export segments ----------------
        rs = 0
        ttk.Label(tab_split, text="把整體剖面（fit-width × fit-height）沿 X 切成 N 段，並在左右各加 M mm 緩衝。").grid(
            row=rs, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        rs += 1
        rs = self._row_text_number(tab_split, rs, "分段數 N（份）", self.var_split_n, hint="例如 6 代表切 6 段；每段核心寬度=fit-width/N")
        rs = self._row_text_number(tab_split, rs, "緩衝距離 M（mm）", self.var_split_buffer_mm, hint="左右各加 M mm；讓曲線不會像硬切（建議 5~20）")
        ttk.Checkbutton(tab_split, text="端點平滑（緩衝區漸進衰減到基準線）", variable=self.var_split_taper).grid(
            row=rs, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        rs += 1
        rs = self._row_text_number(tab_split, rs, "預覽段號（1~N）", self.var_split_index, hint="切換此值可即時預覽該段（含緩衝）")

        ttk.Separator(tab_split).grid(row=rs, column=0, columnspan=2, sticky="ew", pady=10)
        rs += 1
        ttk.Button(tab_split, text="匯出目前分段 DXF…", command=self.on_export_current_segment_dxf).grid(
            row=rs, column=0, columnspan=2, sticky="ew"
        )
        rs += 1
        ttk.Button(tab_split, text="匯出全部分段 DXF（N 個檔）…", command=self.on_export_all_segments_dxf).grid(
            row=rs, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )

        # right preview
        right = ttk.Frame(root)
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(right, background="#111111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # status
        status = ttk.Label(root, textvariable=self.var_status)
        status.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.nb.bind("<<NotebookTabChanged>>", lambda e: self.schedule_update())

    def _row_text_number(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, hint: str) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0, 2))
        ent = ttk.Entry(parent, textvariable=var, width=12)
        ent.grid(row=row, column=1, sticky="w", pady=(0, 2))
        ttk.Label(parent, text=hint, foreground="#666666", wraplength=260, justify="left").grid(
            row=row + 1, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        return row + 2

    def _bind_var_traces(self) -> None:
        # changing these triggers preview update (debounced)
        for v in [
            self.var_sample_step_m,
            self.var_smooth_window_m,
            self.var_simplify_eps_mm,
            self.var_vert_scale,
            self.var_fit_width_mm,
            self.var_fit_height_mm,
            self.var_base_elevation_m,
            self.var_show_profile_normal,
            self.var_show_profile_compressed,
            self.var_compress_gamma,
            self.var_show_grid,
        ]:
            v.trace_add("write", lambda *_: self.schedule_update())

        self.bind("<Configure>", lambda e: self.schedule_update())

    # ---------- Actions ----------
    def on_open(self) -> None:
        path_str = filedialog.askopenfilename(
            title="選擇 KML 或 CSV",
            filetypes=[("KML/CSV", "*.kml *.csv"), ("KML", "*.kml"), ("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path_str:
            return
        self.input_path = Path(path_str)
        self.var_input.set(str(self.input_path))
        self._raw_cache_key = None
        self._raw_samples = None
        self.on_recompute_raw()

    def on_recompute_raw(self) -> None:
        if not self.input_path:
            messagebox.showinfo("提示", "請先開啟一個 KML/CSV 檔案")
            return
        self._compute_raw_async(force=True)

    def on_export_dxf(self) -> None:
        if not self.input_path:
            messagebox.showinfo("提示", "請先開啟一個 KML/CSV 檔案")
            return
        out = filedialog.asksaveasfilename(
            title="匯出 DXF",
            defaultextension=".dxf",
            filetypes=[("DXF", "*.dxf")],
            initialfile="profile.dxf",
        )
        if not out:
            return
        try:
            profiles, meta = self._compute_profiles_for_current_settings()
            gen.write_dxf_polylines(Path(out), profiles)
            messagebox.showinfo(
                "完成",
                f"已匯出 DXF：\n{out}\n\n寬={meta.width_mm:.2f}mm 高={meta.height_mm:.2f}mm | A點數={meta.count_a} | B點數={meta.count_b}",
            )
        except Exception as e:
            messagebox.showerror("錯誤", str(e))

    def on_export_svg(self) -> None:
        if not self.input_path:
            messagebox.showinfo("提示", "請先開啟一個 KML/CSV 檔案")
            return
        out = filedialog.asksaveasfilename(
            title="匯出 SVG",
            defaultextension=".svg",
            filetypes=[("SVG", "*.svg")],
            initialfile="profile.svg",
        )
        if not out:
            return
        try:
            profiles, meta = self._compute_profiles_for_current_settings()
            # SVG: only export Profile A if enabled, otherwise Profile B
            pts = None
            for layer, p in profiles:
                if layer == "PROFILE_A_NORMAL":
                    pts = list(p)
                    break
            if pts is None and profiles:
                pts = list(profiles[0][1])
            if not pts:
                raise ValueError("沒有可匯出的 profile（請至少勾選一種）")
            gen.write_svg_polyline(Path(out), pts)
            messagebox.showinfo("完成", f"已匯出 SVG：\n{out}\n\n寬={meta.width_mm:.2f}mm 高={meta.height_mm:.2f}mm")
        except Exception as e:
            messagebox.showerror("錯誤", str(e))

    def on_export_current_segment_dxf(self) -> None:
        if not self.input_path:
            messagebox.showinfo("提示", "請先開啟一個 KML/CSV 檔案")
            return
        out = filedialog.asksaveasfilename(
            title="匯出目前分段 DXF",
            defaultextension=".dxf",
            filetypes=[("DXF", "*.dxf")],
            initialfile="segment_01.dxf",
        )
        if not out:
            return
        try:
            profiles, _meta = self._compute_split_preview_profiles()
            gen.write_dxf_polylines(Path(out), profiles)
            messagebox.showinfo("完成", f"已匯出：\n{out}")
        except Exception as e:
            messagebox.showerror("錯誤", str(e))

    def on_export_all_segments_dxf(self) -> None:
        if not self.input_path:
            messagebox.showinfo("提示", "請先開啟一個 KML/CSV 檔案")
            return
        folder = filedialog.askdirectory(title="選擇匯出資料夾（會產生 N 個 DXF）")
        if not folder:
            return
        try:
            n = self._parse_int(self.var_split_n.get(), name="分段數 N", min_value=1)
            for i in range(1, n + 1):
                self.var_split_index.set(str(i))
                profiles, _meta = self._compute_split_preview_profiles()
                out_path = Path(folder) / f"segment_{i:02d}.dxf"
                gen.write_dxf_polylines(out_path, profiles)
            messagebox.showinfo("完成", f"已匯出 {n} 個 DXF 到：\n{folder}")
        except Exception as e:
            messagebox.showerror("錯誤", str(e))

    # ---------- Computation ----------
    def schedule_update(self) -> None:
        if self._pending_after_id is not None:
            self.after_cancel(self._pending_after_id)
        self._pending_after_id = self.after(200, self.update_preview)

    def update_preview(self) -> None:
        self._pending_after_id = None
        if not self.input_path:
            self._draw_empty("請先開啟 KML/CSV")
            return

        # ensure raw samples are up-to-date (async recompute if cache key changed)
        try:
            self._compute_raw_async(force=False)
        except Exception as e:
            self.var_status.set(f"參數錯誤：{e}")
            self._draw_empty(f"參數錯誤：{e}")
            return

        if self._raw_samples is None or self._busy:
            self._draw_empty("正在計算/下載海拔…")
            return

        try:
            if hasattr(self, "nb") and self.nb.index(self.nb.select()) == 1:
                profiles, meta = self._compute_split_preview_profiles()
            else:
                profiles, meta = self._compute_profiles_for_current_settings()
            self._draw_profiles(profiles, meta)
        except Exception as e:
            # When users are typing, intermediate values (e.g. "0.") are common; don't crash.
            self.var_status.set(f"參數錯誤：{e}")
            self._draw_empty(f"參數錯誤：{e}")

    def _compute_raw_async(self, force: bool) -> None:
        if self._busy:
            return
        assert self.input_path is not None

        sample_step = self._parse_float(self.var_sample_step_m.get(), name="sample-step-m", min_value=1.0)
        key = RawCacheKey(
            input_path=str(self.input_path),
            input_format="auto",
            sample_step_m=sample_step,
            elevation_source="srtm",
        )
        if (not force) and self._raw_cache_key == key and self._raw_samples is not None:
            return

        self._busy = True
        self.var_status.set("正在計算路徑 / 抽樣海拔（SRTM）…")

        def worker() -> None:
            try:
                samples = self._compute_raw_samples(self.input_path, key.sample_step_m)
                self.after(0, lambda: self._on_raw_done(key, samples, None))
            except Exception as e:
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_raw_done(key, None, f"{e}\n\n{tb}"))

        self._compute_thread = threading.Thread(target=worker, daemon=True)
        self._compute_thread.start()

    def _on_raw_done(self, key: RawCacheKey, samples: Optional[List[gen.Sample]], err: Optional[str]) -> None:
        self._busy = False
        if err:
            self._raw_cache_key = None
            self._raw_samples = None
            self.var_status.set("計算失敗")
            messagebox.showerror("計算失敗", err)
            self._draw_empty("計算失敗")
            return

        assert samples is not None
        self._raw_cache_key = key
        self._raw_samples = samples

        total_km = samples[-1].dist_m / 1000.0
        min_ele = min(s.ele_m for s in samples)
        max_ele = max(s.ele_m for s in samples)
        self.var_status.set(f"完成：{len(samples)} 點 | 長度 {total_km:.2f} km | 海拔 {min_ele:.0f}..{max_ele:.0f} m")
        self.update_preview()

    def _compute_raw_samples(self, path: Path, sample_step_m: float) -> List[gen.Sample]:
        suf = path.suffix.lower()
        if suf == ".csv":
            # Try CSV with elevation first; if missing ele_m, fall back to lat/lon + SRTM
            try:
                return gen.read_samples_from_csv(path)
            except Exception:
                latlon = gen.read_latlon_from_csv(path)
                latlon = gen.densify_latlon_path(latlon, sample_step_m)
                elev = gen.build_elevation_provider("srtm", quiet_download=True)
                return gen.latlon_to_samples(latlon, elev)

        if suf == ".kml":
            latlon = gen.read_latlon_path_from_kml(path)
            latlon = gen.densify_latlon_path(latlon, sample_step_m)
            elev = gen.build_elevation_provider("srtm", quiet_download=True)
            return gen.latlon_to_samples(latlon, elev)

        raise ValueError("只支援 .kml 或 .csv")

    @dataclass
    class Meta:
        width_mm: float
        height_mm: float
        count_a: int
        count_b: int

    def _parse_float(
        self,
        raw: str,
        *,
        name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_empty: bool = False,
        default: Optional[float] = None,
    ) -> float:
        s = (raw or "").strip()
        if s == "":
            if allow_empty and default is not None:
                return float(default)
            if allow_empty:
                raise ValueError(f"{name} 不能為空白")
            raise ValueError(f"{name} 不能為空白")
        try:
            v = float(s)
        except Exception:
            raise ValueError(f"{name} 不是有效數字：{raw!r}")
        if min_value is not None and v < min_value:
            raise ValueError(f"{name} 必須 >= {min_value}")
        if max_value is not None and v > max_value:
            raise ValueError(f"{name} 必須 <= {max_value}")
        return v

    def _parse_int(self, raw: str, *, name: str, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
        s = (raw or "").strip()
        if s == "":
            raise ValueError(f"{name} 不能為空白")
        try:
            v = int(float(s))
        except Exception:
            raise ValueError(f"{name} 不是有效整數：{raw!r}")
        if min_value is not None and v < min_value:
            raise ValueError(f"{name} 必須 >= {min_value}")
        if max_value is not None and v > max_value:
            raise ValueError(f"{name} 必須 <= {max_value}")
        return v

    def _parse_optional_float(self, raw: str, *, name: str) -> Optional[float]:
        s = (raw or "").strip()
        if s == "":
            return None
        try:
            return float(s)
        except Exception:
            raise ValueError(f"{name} 不是有效數字：{raw!r}")

    def _compute_profiles_for_current_settings(self) -> Tuple[List[Tuple[str, List[Point]]], "ProfileGUI.Meta"]:
        assert self._raw_samples is not None

        samples: Sequence[gen.Sample] = self._raw_samples
        smooth_m = self._parse_float(self.var_smooth_window_m.get(), name="smooth-window-m", min_value=0.0)
        if smooth_m > 0:
            samples = gen.smooth_elevation_by_distance(samples, window_m=smooth_m)

        base_ele = self._parse_optional_float(self.var_base_elevation_m.get(), name="base-elevation-m")

        vert_scale = self._parse_float(self.var_vert_scale.get(), name="vert-scale", min_value=0.0001)
        fit_w = self._parse_float(self.var_fit_width_mm.get(), name="fit-width(mm)", min_value=1.0)
        fit_h = self._parse_float(self.var_fit_height_mm.get(), name="fit-height(mm)", min_value=1.0)
        eps = self._parse_float(self.var_simplify_eps_mm.get(), name="simplify-epsilon(mm)", min_value=0.0)
        gamma = self._parse_float(self.var_compress_gamma.get(), name="compress-gamma", min_value=0.0001, max_value=1.0)

        if not self.var_show_profile_normal.get() and not self.var_show_profile_compressed.get():
            raise ValueError("請至少勾選一種 profile（原始 / 高度壓縮）")

        # Profile A (normal)
        pts_a = gen.samples_to_profile_points(
            samples=samples,
            out_unit="mm",
            horiz_scale=1.0,
            vert_scale=vert_scale,
            base_elevation_m=base_ele,
        )

        # Profile B (compressed)
        pts_b: Optional[List[Point]] = None
        if self.var_show_profile_compressed.get():
            comp_samples = gen.compress_elevation_range(samples, base_elevation_m=base_ele, gamma=gamma)
            pts_b = gen.samples_to_profile_points(
                samples=comp_samples,
                out_unit="mm",
                horiz_scale=1.0,
                vert_scale=vert_scale,
                base_elevation_m=base_ele,
            )

        # Fit based on Profile A bbox (so A/B are comparable on same scale)
        minx, miny, maxx, maxy = gen.bbox(pts_a)
        w = maxx - minx
        h = maxy - miny

        sx = 1.0
        sy = 1.0
        if w > 0:
            sx = fit_w / w
        if h > 0:
            sy = fit_h / h
        if sx != 1.0 or sy != 1.0:
            pts_a = gen.scale_points(pts_a, sx=sx, sy=sy)
            if pts_b is not None:
                pts_b = gen.scale_points(pts_b, sx=sx, sy=sy)

        if eps > 0 and len(pts_a) >= 3:
            pts_a = gen.simplify_rdp(pts_a, eps)
        if pts_b is not None and eps > 0 and len(pts_b) >= 3:
            pts_b = gen.simplify_rdp(pts_b, eps)

        minx2, miny2, maxx2, maxy2 = gen.bbox(pts_a)
        profiles: List[Tuple[str, List[Point]]] = []
        if self.var_show_profile_normal.get():
            profiles.append(("PROFILE_A_NORMAL", list(pts_a)))
        if pts_b is not None:
            profiles.append(("PROFILE_B_COMPRESSED", list(pts_b)))

        meta = self.Meta(
            width_mm=(maxx2 - minx2),
            height_mm=(maxy2 - miny2),
            count_a=len(pts_a),
            count_b=0 if pts_b is None else len(pts_b),
        )
        return profiles, meta

    # ---------- Split / segment helpers ----------
    def _clip_polyline_x(self, pts: Sequence[Point], xmin: float, xmax: float) -> List[Point]:
        """Clip a polyline (monotonic in x) to [xmin, xmax], inserting boundary points by linear interpolation."""
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        out: List[Point] = []
        if not pts:
            return out

        def interp(p1: Point, p2: Point, x: float) -> Point:
            (x1, y1) = p1
            (x2, y2) = p2
            if x2 == x1:
                return (x, y2)
            t = (x - x1) / (x2 - x1)
            return (x, y1 + (y2 - y1) * t)

        for (p1, p2) in zip(pts, pts[1:]):
            x1, y1 = p1
            x2, y2 = p2
            # segment entirely outside
            if x2 < xmin or x1 > xmax:
                continue
            # entering
            if x1 < xmin <= x2:
                out.append(interp(p1, p2, xmin))
            # interior point
            if xmin <= x2 <= xmax:
                out.append(p2)
            # leaving
            if x1 <= xmax < x2:
                out.append(interp(p1, p2, xmax))
                break

        # handle case where first point already inside but loop didn't add it
        if pts and xmin <= pts[0][0] <= xmax:
            if not out or out[0][0] != pts[0][0]:
                out.insert(0, pts[0])

        # ensure starts/ends at boundaries if possible
        if out:
            if out[0][0] < xmin - 1e-6:
                out[0] = (xmin, out[0][1])
            if out[-1][0] > xmax + 1e-6:
                out[-1] = (xmax, out[-1][1])
        return out

    def _taper_to_baseline(self, pts: Sequence[Point], *, x0: float, x1: float, buffer_mm: float) -> List[Point]:
        """Apply smooth taper in buffer zones so y fades to 0 at ends."""
        if buffer_mm <= 0:
            return list(pts)

        def smoothstep(t: float) -> float:
            t = max(0.0, min(1.0, t))
            return t * t * (3 - 2 * t)

        xmin = x0 - buffer_mm
        xmax = x1 + buffer_mm
        out: List[Point] = []
        for x, y in pts:
            w = 1.0
            if x < x0:
                w = smoothstep((x - xmin) / buffer_mm)
            elif x > x1:
                w = smoothstep((xmax - x) / buffer_mm)
            out.append((x, y * w))
        return out

    def _shift_x0(self, pts: Sequence[Point]) -> List[Point]:
        if not pts:
            return []
        x0 = pts[0][0]
        return [(x - x0, y) for (x, y) in pts]

    def _compute_split_preview_profiles(self) -> Tuple[List[Tuple[str, List[Point]]], "ProfileGUI.Meta"]:
        # Start from the full profiles (already fit to fit-width/fit-height)
        full_profiles, full_meta = self._compute_profiles_for_current_settings()

        fit_w = self._parse_float(self.var_fit_width_mm.get(), name="fit-width(mm)", min_value=1.0)
        n = self._parse_int(self.var_split_n.get(), name="分段數 N", min_value=1)
        idx = self._parse_int(self.var_split_index.get(), name="預覽段號", min_value=1, max_value=n)
        buffer_mm = self._parse_float(self.var_split_buffer_mm.get(), name="緩衝距離 M(mm)", min_value=0.0)

        core_w = fit_w / n
        x0 = (idx - 1) * core_w
        x1 = idx * core_w
        xmin = max(0.0, x0 - buffer_mm)
        xmax = min(fit_w, x1 + buffer_mm)

        out_profiles: List[Tuple[str, List[Point]]] = []
        for layer, pts in full_profiles:
            clipped = self._clip_polyline_x(pts, xmin, xmax)
            if self.var_split_taper.get():
                clipped = self._taper_to_baseline(clipped, x0=x0, x1=x1, buffer_mm=buffer_mm)
            clipped = self._shift_x0(clipped)
            out_profiles.append((layer, clipped))

        # meta: reuse height from full meta, width becomes clipped width (approx)
        width_mm = (xmax - xmin)
        meta = self.Meta(width_mm=width_mm, height_mm=full_meta.height_mm, count_a=full_meta.count_a, count_b=full_meta.count_b)
        return out_profiles, meta

    # ---------- Drawing ----------
    def _draw_empty(self, text: str) -> None:
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.canvas.create_text(w // 2, h // 2, text=text, fill="#bbbbbb", font=("Segoe UI", 14))

    def _draw_profiles(self, profiles: Sequence[Tuple[str, Sequence[Point]]], meta: "ProfileGUI.Meta") -> None:
        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        # bbox based on Profile A if present, else first profile
        if not profiles:
            self._draw_empty("沒有可預覽的 profile")
            return
        points_a = None
        for layer, pts in profiles:
            if layer == "PROFILE_A_NORMAL":
                points_a = pts
                break
        base_pts = points_a if points_a is not None else profiles[0][1]

        minx, miny, maxx, maxy = gen.bbox(base_pts)
        w = max(1e-9, maxx - minx)
        h = max(1e-9, maxy - miny)

        pad = 30
        sx = (cw - 2 * pad) / w
        sy = (ch - 2 * pad) / h
        s = min(sx, sy)

        # optional grid (in mm)
        if self.var_show_grid.get():
            grid_mm = self._choose_grid_step(max(meta.width_mm, meta.height_mm))
            self._draw_grid(minx, miny, maxx, maxy, s, pad, grid_mm)

        # draw each profile with different color
        colors = {
            "PROFILE_A_NORMAL": "#00e5ff",
            "PROFILE_B_COMPRESSED": "#ff4dff",
        }
        for layer, pts in profiles:
            coords: List[float] = []
            for x, y in pts:
                px = pad + (x - minx) * s
                py = ch - pad - (y - miny) * s
                coords.extend([px, py])
            self.canvas.create_line(*coords, fill=colors.get(layer, "#ffffff"), width=2, smooth=False)

        # overlay info
        info = f"預覽：寬={meta.width_mm:.1f}mm 高={meta.height_mm:.1f}mm | A點數={meta.count_a} | B點數={meta.count_b}"
        self.canvas.create_rectangle(10, 10, 10 + 640, 40, fill="#000000", outline="")
        self.canvas.create_text(20, 25, text=info, anchor="w", fill="#ffffff", font=("Segoe UI", 11))

    def _choose_grid_step(self, max_dim_mm: float) -> float:
        # choose a step so ~10-20 lines across
        for step in [1, 2, 5, 10, 20, 25, 50, 100, 200]:
            if max_dim_mm / step <= 20:
                return float(step)
        return 500.0

    def _draw_grid(self, minx: float, miny: float, maxx: float, maxy: float, s: float, pad: int, step_mm: float) -> None:
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        # vertical lines
        x = (minx // step_mm) * step_mm
        while x <= maxx:
            px = pad + (x - minx) * s
            self.canvas.create_line(px, pad, px, ch - pad, fill="#222222")
            x += step_mm
        # horizontal lines
        y = (miny // step_mm) * step_mm
        while y <= maxy:
            py = ch - pad - (y - miny) * s
            self.canvas.create_line(pad, py, cw - pad, py, fill="#222222")
            y += step_mm


def main() -> None:
    app = ProfileGUI()
    app.mainloop()


if __name__ == "__main__":
    main()



