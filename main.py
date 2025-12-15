"""
Taiwan N-S profile path generator for Autodesk Fusion 360.

Goal:
  - Convert a route's elevation samples into a 2D profile polyline
    (X = cumulative distance, Y = elevation) and export as DXF or SVG.
  - Import the resulting file into Fusion 360 (Insert DXF / Insert SVG)
    to build 3D models (extrude, loft, sweep, etc.).

Input formats (currently):
  - CSV with columns: lat, lon, ele_m
  - CSV with columns: lat, lon              (ele will be sampled from SRTM if enabled)
  - CSV with columns: dist_m, ele_m (pre-computed distance)
  - KML path (.kml)                         (ele will be sampled from SRTM)

Output:
  - DXF (R2010) containing a single LWPOLYLINE on layer "PROFILE"
  - (Optional) SVG polyline
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Sample:
    dist_m: float
    ele_m: float


def configure_logging(verbose: bool) -> None:
    """
    Keep output clean by default.
    - ezdxf/fontTools can emit many font warnings on Windows even when we don't use TEXT entities.
    - Some DEM download helpers print progress to stdout/stderr.
    """
    # Don't override user's logging handlers if they already configured it.
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")

    if not verbose:
        logging.getLogger("ezdxf").setLevel(logging.ERROR)
        logging.getLogger("fontTools").setLevel(logging.ERROR)


@contextlib.contextmanager
def suppress_output(enabled: bool) -> Iterable[None]:
    """Suppress stdout/stderr for noisy third-party libs (enabled=True)."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as dn:
        with contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            yield


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    r = 6371008.8  # mean earth radius (m)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _parse_float(value: str) -> float:
    v = value.strip()
    if v == "":
        raise ValueError("empty numeric value")
    return float(v)


def read_samples_from_csv(path: Path) -> List[Sample]:
    """
    Supports:
      - lat, lon, ele_m  -> computes cumulative distance
      - dist_m, ele_m    -> uses provided distance
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        fields = {name.strip().lower(): name for name in reader.fieldnames}

        def has(*names: str) -> bool:
            return all(n in fields for n in names)

        if has("dist_m", "ele_m"):
            out: List[Sample] = []
            for i, row in enumerate(reader, start=2):
                try:
                    dist = _parse_float(row[fields["dist_m"]])
                    ele = _parse_float(row[fields["ele_m"]])
                except Exception as e:
                    raise ValueError(f"CSV parse error at line {i}: {e}") from e
                out.append(Sample(dist_m=dist, ele_m=ele))
            return _normalize_distance(out)

        if has("lat", "lon", "ele_m"):
            latlon: List[Tuple[float, float, float]] = []
            for i, row in enumerate(reader, start=2):
                try:
                    lat = _parse_float(row[fields["lat"]])
                    lon = _parse_float(row[fields["lon"]])
                    ele = _parse_float(row[fields["ele_m"]])
                except Exception as e:
                    raise ValueError(f"CSV parse error at line {i}: {e}") from e
                latlon.append((lat, lon, ele))

            out2: List[Sample] = []
            dist = 0.0
            prev = None
            for lat, lon, ele in latlon:
                if prev is not None:
                    dist += haversine_m(prev[0], prev[1], lat, lon)
                out2.append(Sample(dist_m=dist, ele_m=ele))
                prev = (lat, lon)
            return _normalize_distance(out2)

        if has("lat", "lon"):
            # lat/lon only -> caller can sample elevations (e.g. SRTM)
            latlon2: List[Tuple[float, float]] = []
            for i, row in enumerate(reader, start=2):
                try:
                    lat = _parse_float(row[fields["lat"]])
                    lon = _parse_float(row[fields["lon"]])
                except Exception as e:
                    raise ValueError(f"CSV parse error at line {i}: {e}") from e
                latlon2.append((lat, lon))
            raise ValueError(
                "CSV has lat/lon but no ele_m. Use --elevation-source srtm to sample elevations, "
                "and use --input-format csv-latlon if auto-detection didn't work."
            )

        raise ValueError(
            "Unsupported CSV schema. Expected headers:\n"
            "  - lat, lon, ele_m\n"
            "  - lat, lon (use --elevation-source srtm)\n"
            "  - dist_m, ele_m"
        )


def read_latlon_from_csv(path: Path) -> List[Tuple[float, float]]:
    """Read CSV lat/lon without elevation. Headers: lat, lon"""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        if "lat" not in fields or "lon" not in fields:
            raise ValueError("CSV must include headers: lat, lon")
        pts: List[Tuple[float, float]] = []
        for i, row in enumerate(reader, start=2):
            try:
                lat = _parse_float(row[fields["lat"]])
                lon = _parse_float(row[fields["lon"]])
            except Exception as e:
                raise ValueError(f"CSV parse error at line {i}: {e}") from e
            pts.append((lat, lon))
        if not pts:
            raise ValueError("No lat/lon points found.")
        return pts


def read_latlon_path_from_kml(path: Path) -> List[Tuple[float, float]]:
    """
    Parse the first <coordinates> block in KML.
    Coordinates are typically: lon,lat[,alt] separated by spaces/newlines.
    """
    import xml.etree.ElementTree as ET

    text = path.read_text(encoding="utf-8", errors="ignore")
    # ElementTree doesn't like some KML headers; parse from string is generally OK.
    root = ET.fromstring(text)

    # KML uses namespaces. We'll match any tag ending with 'coordinates'.
    coords_elems = [e for e in root.iter() if (e.tag.endswith("coordinates"))]
    if not coords_elems:
        raise ValueError("KML has no <coordinates> element.")

    raw = (coords_elems[0].text or "").strip()
    if not raw:
        raise ValueError("KML <coordinates> is empty.")

    pts: List[Tuple[float, float]] = []
    for token in raw.replace("\n", " ").replace("\t", " ").split(" "):
        t = token.strip()
        if not t:
            continue
        parts = t.split(",")
        if len(parts) < 2:
            continue
        lon = float(parts[0])
        lat = float(parts[1])
        pts.append((lat, lon))

    if len(pts) < 2:
        raise ValueError("KML coordinates parsed < 2 points; please export a Path with multiple vertices.")
    return pts


def densify_latlon_path(points: Sequence[Tuple[float, float]], step_m: float) -> List[Tuple[float, float]]:
    """Insert intermediate lat/lon points so adjacent spacing is ~ step_m."""
    if step_m <= 0:
        return list(points)
    if len(points) < 2:
        return list(points)

    out: List[Tuple[float, float]] = [points[0]]
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        seg = haversine_m(lat1, lon1, lat2, lon2)
        if seg <= step_m:
            out.append((lat2, lon2))
            continue
        n = int(math.ceil(seg / step_m))
        for i in range(1, n + 1):
            t = i / n
            lat = lat1 + (lat2 - lat1) * t
            lon = lon1 + (lon2 - lon1) * t
            out.append((lat, lon))
    return out


def fill_nan_linear(values: List[float]) -> List[float]:
    """Fill NaNs by linear interpolation; edge NaNs are forward/back filled."""
    if not values:
        return values
    isnan = [math.isnan(v) for v in values]
    if all(isnan):
        raise ValueError("All sampled elevations are missing (NaN).")

    out = values[:]
    # forward fill leading NaNs
    first = next(i for i, v in enumerate(out) if not math.isnan(v))
    for i in range(0, first):
        out[i] = out[first]
    # back fill trailing NaNs
    last = len(out) - 1 - next(i for i, v in enumerate(reversed(out)) if not math.isnan(v))
    for i in range(last + 1, len(out)):
        out[i] = out[last]

    i = 0
    while i < len(out):
        if not math.isnan(out[i]):
            i += 1
            continue
        j = i
        while j < len(out) and math.isnan(out[j]):
            j += 1
        # now [i, j) are NaNs, interpolate between out[i-1] and out[j]
        left = out[i - 1]
        right = out[j]
        span = j - (i - 1)
        for k in range(i, j):
            t = (k - (i - 1)) / span
            out[k] = left + (right - left) * t
        i = j
    return out


def smooth_elevation_by_distance(samples: Sequence[Sample], window_m: float) -> List[Sample]:
    """
    Smooth elevations with a centered moving average over a distance window.
    window_m: total window width in meters along the route (e.g., 2000 = +/- 1000m).
    """
    if window_m <= 0:
        return list(samples)
    if len(samples) < 3:
        return list(samples)

    half = window_m / 2.0
    dists = [s.dist_m for s in samples]
    eles = [s.ele_m for s in samples]

    out_ele: List[float] = [0.0] * len(samples)
    left = 0
    right = 0
    acc = 0.0
    count = 0

    for i in range(len(samples)):
        center = dists[i]

        while right < len(samples) and dists[right] <= center + half:
            acc += eles[right]
            count += 1
            right += 1

        while left < len(samples) and dists[left] < center - half:
            acc -= eles[left]
            count -= 1
            left += 1

        out_ele[i] = eles[i] if count <= 0 else (acc / count)

    return [Sample(dist_m=s.dist_m, ele_m=e) for s, e in zip(samples, out_ele)]


def build_elevation_provider(source: str, *, quiet_download: bool) -> Callable[[float, float], float]:
    """
    Returns a function (lat, lon) -> elevation meters.
    Currently supports: srtm
    """
    src = source.lower().strip()
    if src == "srtm":
        try:
            import srtm  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing dependency for SRTM. Run: pip install srtm.py") from e

        # Some environments print progress during tile downloads; keep it quiet unless verbose.
        with suppress_output(enabled=quiet_download):
            data = srtm.get_data()
        cache: dict[Tuple[int, int], float] = {}

        def get(lat: float, lon: float) -> float:
            # cache by 1e-4 deg (~11m) bucket to reduce repeated lookups
            key = (int(round(lat * 1e4)), int(round(lon * 1e4)))
            if key in cache:
                return cache[key]
            v = data.get_elevation(lat, lon)
            out = float("nan") if v is None else float(v)
            cache[key] = out
            return out

        return get
    raise ValueError("elevation source must be: srtm")


def latlon_to_samples(
    points: Sequence[Tuple[float, float]],
    elevation_provider: Callable[[float, float], float],
) -> List[Sample]:
    """Compute cumulative distance and sample elevations."""
    if len(points) < 2:
        raise ValueError("Need at least 2 lat/lon points.")
    dists: List[float] = [0.0]
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        dists.append(dists[-1] + haversine_m(lat1, lon1, lat2, lon2))
    eles = [elevation_provider(lat, lon) for lat, lon in points]
    eles = fill_nan_linear(eles)
    return _normalize_distance([Sample(dist_m=d, ele_m=e) for d, e in zip(dists, eles)])


def _normalize_distance(samples: Sequence[Sample]) -> List[Sample]:
    """Shift distance so the first point starts at 0."""
    if not samples:
        raise ValueError("No samples found.")
    d0 = samples[0].dist_m
    return [Sample(dist_m=s.dist_m - d0, ele_m=s.ele_m) for s in samples]


def simplify_rdp(points: Sequence[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
    """
    Ramer–Douglas–Peucker polyline simplification.
    epsilon: in same units as points (e.g., mm if you convert to mm first)
    """
    if len(points) < 3:
        return list(points)

    def perpendicular_distance(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        (x, y) = p
        (x1, y1) = a
        (x2, y2) = b
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(x - x1, y - y1)
        t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        proj = (x1 + t * dx, y1 + t * dy)
        return math.hypot(x - proj[0], y - proj[1])

    a = points[0]
    b = points[-1]
    max_dist = -1.0
    idx = -1
    for i in range(1, len(points) - 1):
        d = perpendicular_distance(points[i], a, b)
        if d > max_dist:
            max_dist = d
            idx = i

    if max_dist > epsilon:
        left = simplify_rdp(points[: idx + 1], epsilon)
        right = simplify_rdp(points[idx:], epsilon)
        return left[:-1] + right
    return [a, b]


def samples_to_profile_points(
    samples: Sequence[Sample],
    out_unit: str,
    horiz_scale: float,
    vert_scale: float,
    base_elevation_m: Optional[float],
) -> List[Tuple[float, float]]:
    """
    Convert (dist_m, ele_m) to 2D points in output units.

    out_unit:
      - "mm" or "cm" or "m"
    horiz_scale:
      - multiply horizontal distance (after converting to output units)
    vert_scale:
      - multiply vertical distance (after converting to output units)
    base_elevation_m:
      - if provided, Y = (ele_m - base_elevation_m)
      - if None, uses min(ele_m) as base to make profile start at Y>=0
    """
    if out_unit not in {"mm", "cm", "m"}:
        raise ValueError("out_unit must be one of: mm, cm, m")

    m_to_unit = {"m": 1.0, "cm": 100.0, "mm": 1000.0}[out_unit]
    eles = [s.ele_m for s in samples]
    base = min(eles) if base_elevation_m is None else base_elevation_m

    pts: List[Tuple[float, float]] = []
    for s in samples:
        x = (s.dist_m * m_to_unit) * horiz_scale
        y = ((s.ele_m - base) * m_to_unit) * vert_scale
        pts.append((x, y))
    return pts


def compress_elevation_range(
    samples: Sequence[Sample],
    *,
    base_elevation_m: Optional[float],
    gamma: float,
) -> List[Sample]:
    """
    Nonlinear elevation compression to reduce the height difference between low/high mountains.

    We map delta = max(0, ele - base) into:
      delta' = (delta / delta_max) ** gamma * delta_max
    where 0 < gamma <= 1.

    - gamma = 1.0: unchanged
    - gamma < 1.0: compress high peaks, lift lower parts relatively (smaller contrast)
    """
    if not samples:
        raise ValueError("No samples to compress.")
    if gamma <= 0 or gamma > 1.0:
        raise ValueError("gamma must be in (0, 1].")

    eles = [s.ele_m for s in samples]
    base = min(eles) if base_elevation_m is None else float(base_elevation_m)
    deltas = [max(0.0, e - base) for e in eles]
    dmax = max(deltas)
    if dmax <= 0:
        return list(samples)

    out: List[Sample] = []
    for s, d in zip(samples, deltas):
        dn = d / dmax
        d2 = (dn ** gamma) * dmax
        out.append(Sample(dist_m=s.dist_m, ele_m=base + d2))
    return out


def write_dxf_polylines(path: Path, polylines: Sequence[Tuple[str, Sequence[Tuple[float, float]]]]) -> None:
    """
    Write multiple 2D polylines into one DXF, each on its own layer.
    polylines: [(layer_name, points), ...]
    """
    if not polylines:
        raise ValueError("No polylines to write.")
    for layer, pts in polylines:
        if not pts:
            raise ValueError(f"Polyline '{layer}' has no points.")

    try:
        import ezdxf  # type: ignore
    except Exception:
        ezdxf = None

    if ezdxf is not None:
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        for layer, pts in polylines:
            if layer not in doc.layers:
                doc.layers.new(layer, dxfattribs={"color": 7})
            msp.add_lwpolyline(list(pts), dxfattribs={"layer": layer})
        doc.saveas(str(path))
        return

    # Fallback: very small DXF R12 writer with multiple POLYLINE blocks
    def w(lines: List[str], code: str, value: str) -> None:
        lines.append(code)
        lines.append(value)

    lines: List[str] = []
    w(lines, "0", "SECTION")
    w(lines, "2", "HEADER")
    w(lines, "0", "ENDSEC")
    w(lines, "0", "SECTION")
    w(lines, "2", "TABLES")
    w(lines, "0", "ENDSEC")
    w(lines, "0", "SECTION")
    w(lines, "2", "ENTITIES")
    for layer, pts in polylines:
        w(lines, "0", "POLYLINE")
        w(lines, "8", layer)
        w(lines, "66", "1")
        w(lines, "70", "0")
        for x, y in pts:
            w(lines, "0", "VERTEX")
            w(lines, "8", layer)
            w(lines, "10", f"{x:.6f}")
            w(lines, "20", f"{y:.6f}")
            w(lines, "30", "0.0")
        w(lines, "0", "SEQEND")
    w(lines, "0", "ENDSEC")
    w(lines, "0", "EOF")
    path.write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")


def bbox(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy)."""
    if not points:
        raise ValueError("No points for bbox.")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def scale_points(points: Sequence[Tuple[float, float]], sx: float, sy: float) -> List[Tuple[float, float]]:
    return [(x * sx, y * sy) for (x, y) in points]


def write_svg_polyline(path: Path, points: Sequence[Tuple[float, float]]) -> None:
    if not points:
        raise ValueError("No points to write.")
    minx, miny, maxx, maxy = bbox(points)
    w = max(1.0, maxx - minx)
    h = max(1.0, maxy - miny)

    # SVG Y axis goes down, flip it so the profile is "up" visually.
    def fmt(p: Tuple[float, float]) -> str:
        x = p[0] - minx
        y = (maxy - p[1])
        return f"{x:.3f},{y:.3f}"

    pts = " ".join(fmt(p) for p in points)
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w:.3f}" height="{h:.3f}" viewBox="0 0 {w:.3f} {h:.3f}">
  <polyline points="{pts}" fill="none" stroke="black" stroke-width="1"/>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_dxf_lwpolyline(path: Path, points: Sequence[Tuple[float, float]]) -> None:
    """
    Minimal DXF writer using ezdxf if available; falls back to a very small R12 writer.
    Fusion 360 generally imports both, but ezdxf output is safer.
    """
    if not points:
        raise ValueError("No points to write.")

    try:
        import ezdxf  # type: ignore
    except Exception:
        ezdxf = None

    if ezdxf is not None:
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        doc.layers.new("PROFILE", dxfattribs={"color": 7})
        msp.add_lwpolyline(points, dxfattribs={"layer": "PROFILE"})
        doc.saveas(str(path))
        return

    # Fallback: extremely small DXF R12 POLYLINE (2D) writer
    # NOTE: This is intentionally minimal; ezdxf is recommended.
    def w(lines: List[str], code: str, value: str) -> None:
        lines.append(code)
        lines.append(value)

    lines: List[str] = []
    w(lines, "0", "SECTION")
    w(lines, "2", "HEADER")
    w(lines, "0", "ENDSEC")
    w(lines, "0", "SECTION")
    w(lines, "2", "TABLES")
    w(lines, "0", "ENDSEC")
    w(lines, "0", "SECTION")
    w(lines, "2", "ENTITIES")

    w(lines, "0", "POLYLINE")
    w(lines, "8", "PROFILE")
    w(lines, "66", "1")  # vertices follow
    w(lines, "70", "0")  # open polyline
    for x, y in points:
        w(lines, "0", "VERTEX")
        w(lines, "8", "PROFILE")
        w(lines, "10", f"{x:.6f}")
        w(lines, "20", f"{y:.6f}")
        w(lines, "30", "0.0")
    w(lines, "0", "SEQEND")
    w(lines, "0", "ENDSEC")
    w(lines, "0", "EOF")
    path.write_text("\n".join(lines) + "\n", encoding="ascii", errors="ignore")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a Fusion 360-importable profile polyline (DXF/SVG) from CSV samples."
    )
    p.add_argument("--input", "-i", required=True, help="Input path (.csv or .kml).")
    p.add_argument("--output", "-o", required=True, help="Output path (.dxf or .svg).")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show third-party warnings/progress output (ezdxf/fontTools/SRTM download).",
    )
    p.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "csv", "csv-latlon", "kml"],
        help="How to interpret input. auto uses file extension.",
    )
    p.add_argument(
        "--elevation-source",
        default="srtm",
        choices=["srtm"],
        help="Elevation source for lat/lon-only inputs (KML or CSV without ele_m).",
    )
    p.add_argument(
        "--no-quiet-download",
        dest="quiet_download",
        action="store_false",
        default=True,
        help="Show DEM download progress output (default is quiet).",
    )
    p.add_argument(
        "--sample-step-m",
        type=float,
        default=200.0,
        help="Densify lat/lon path to this spacing (meters) before sampling elevations.",
    )
    p.add_argument("--out-unit", default="mm", choices=["mm", "cm", "m"], help="Output units.")
    p.add_argument("--horiz-scale", type=float, default=1.0, help="Horizontal scale multiplier.")
    p.add_argument("--vert-scale", type=float, default=1.0, help="Vertical scale multiplier (vertical exaggeration).")
    p.add_argument(
        "--smooth-window-m",
        type=float,
        default=0.0,
        help="Smooth elevations with a moving average window in meters along the route (e.g., 2000). 0 disables.",
    )
    p.add_argument(
        "--fit-width",
        type=float,
        default=None,
        help="Auto-scale X so the profile width fits this value (in --out-unit). Example: --fit-width 240",
    )
    p.add_argument(
        "--fit-height",
        type=float,
        default=None,
        help="Auto-scale Y so the profile height fits this value (in --out-unit).",
    )
    p.add_argument(
        "--fit-box",
        type=float,
        default=None,
        help="Uniformly scale X/Y so max(width,height) fits this value (in --out-unit). Example: --fit-box 240",
    )
    p.add_argument(
        "--base-elevation-m",
        type=float,
        default=None,
        help="Base elevation (m). If omitted, uses min elevation so the profile starts at Y=0.",
    )
    p.add_argument(
        "--simplify-epsilon",
        type=float,
        default=0.0,
        help="RDP simplify epsilon in output units (e.g., mm). 0 disables simplification.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=bool(getattr(args, "verbose", False)))
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    fmt = args.input_format
    if fmt == "auto":
        suf = in_path.suffix.lower()
        if suf == ".kml":
            fmt = "kml"
        elif suf == ".csv":
            fmt = "csv"
        else:
            print("ERROR: cannot auto-detect input format. Use --input-format csv|csv-latlon|kml", file=sys.stderr)
            return 2

    if fmt == "csv":
        # If CSV is lat/lon only, tell user to use csv-latlon mode.
        samples = read_samples_from_csv(in_path)
    elif fmt == "csv-latlon":
        latlon = read_latlon_from_csv(in_path)
        latlon = densify_latlon_path(latlon, args.sample_step_m)
        elev = build_elevation_provider(args.elevation_source, quiet_download=bool(args.quiet_download) and (not args.verbose))
        samples = latlon_to_samples(latlon, elev)
    elif fmt == "kml":
        latlon = read_latlon_path_from_kml(in_path)
        latlon = densify_latlon_path(latlon, args.sample_step_m)
        elev = build_elevation_provider(args.elevation_source, quiet_download=bool(args.quiet_download) and (not args.verbose))
        samples = latlon_to_samples(latlon, elev)
    else:
        print("ERROR: unsupported --input-format", file=sys.stderr)
        return 2

    if args.smooth_window_m and args.smooth_window_m > 0:
        samples = smooth_elevation_by_distance(samples, window_m=float(args.smooth_window_m))

    points = samples_to_profile_points(
        samples=samples,
        out_unit=args.out_unit,
        horiz_scale=args.horiz_scale,
        vert_scale=args.vert_scale,
        base_elevation_m=args.base_elevation_m,
    )

    # Auto-fit to target dimensions (in output units) if requested.
    minx, miny, maxx, maxy = bbox(points)
    w = maxx - minx
    h = maxy - miny
    sx = 1.0
    sy = 1.0
    if args.fit_box is not None:
        if args.fit_box <= 0:
            print("ERROR: --fit-box must be > 0", file=sys.stderr)
            return 2
        m = max(w, h)
        if m > 0:
            s = args.fit_box / m
            sx = s
            sy = s
    if args.fit_width is not None:
        if args.fit_width <= 0:
            print("ERROR: --fit-width must be > 0", file=sys.stderr)
            return 2
        if w > 0 and args.fit_box is None:
            sx = args.fit_width / w
    if args.fit_height is not None:
        if args.fit_height <= 0:
            print("ERROR: --fit-height must be > 0", file=sys.stderr)
            return 2
        if h > 0 and args.fit_box is None:
            sy = args.fit_height / h
    if sx != 1.0 or sy != 1.0:
        points = scale_points(points, sx=sx, sy=sy)

    if args.simplify_epsilon and args.simplify_epsilon > 0:
        points = simplify_rdp(points, args.simplify_epsilon)

    suffix = out_path.suffix.lower()
    if suffix == ".dxf":
        write_dxf_lwpolyline(out_path, points)
    elif suffix == ".svg":
        write_svg_polyline(out_path, points)
    else:
        print("ERROR: output must end with .dxf or .svg", file=sys.stderr)
        return 2

    total_km = samples[-1].dist_m / 1000.0
    min_ele = min(s.ele_m for s in samples)
    max_ele = max(s.ele_m for s in samples)
    print(f"OK: wrote {out_path}")
    print(f"Samples: {len(samples)} | Length: {total_km:.2f} km | Elev: {min_ele:.1f}..{max_ele:.1f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

