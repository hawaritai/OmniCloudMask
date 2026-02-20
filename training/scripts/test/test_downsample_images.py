"""
Aerial Imagery Downsampler
--------------------------
Reads a project config, computes target resolution from GSD ratio,
and batch-converts images to the desired resolution in a separate output folder.

Image reading strategy
~~~~~~~~~~~~~~~~~~~~~~
- IIQ (Phase One raw)  → rawpy demosaics the full raw sensor data (NOT the
                          embedded preview) to a 16-bit sRGB numpy array.
- TIF / JPG / anything  → rasterio reads natively; CRS and geotransform are
                          preserved when present.

Downsampling & saving
~~~~~~~~~~~~~~~~~~~~~
Resampling is done with PIL LANCZOS applied directly on the in-memory numpy
array.  The old rasterio.MemoryFile path is gone -- it was a double-buffer
(array -> MemoryFile copy -> resample) that consumed 2x RAM per worker and
caused _tiffWriteProc "Not enough space" errors under parallel load.

Output is always a lossless GeoTIFF:
    DEFLATE compression + predictor=2 (horizontal differencing for integers)
    Tiled 512x512, ZLEVEL 9 -- maximum lossless compression, zero quality loss.

Worker count
~~~~~~~~~~~~
Workers default to  min(cpu_count - 1,  available_ram / (image_bytes * RAM_FACTOR))
so you never exhaust RAM regardless of core count.  Pass --workers N to override.
The auto value is enforced as a hard RAM cap even when you supply --workers.

Usage
~~~~~
    python downsample_aerial.py --project 2025_08_10_AIR_PHKIO
    python downsample_aerial.py --project all
    python downsample_aerial.py --project all --extra-gsd 1000
    python downsample_aerial.py --list
    python downsample_aerial.py --workers 8     # explicit worker count
"""

import argparse
import logging
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import rasterio
from PIL import Image
from rasterio.transform import Affine

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("aerial_downsample")

for _lib in ("rasterio", "fiona", "rawpy", "PIL"):
    logging.getLogger(_lib).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------
PROJECT_CONFIG: dict[str, dict] = {
    # "2051_102025082_D82_Tarn-et-Garonne_A": {
    #     "image_dir":   r"Q:\02_PROJECTS\2051_102025082_D82_Tarn-et-Garonne_A\60_UM\LVL03_CertiflAI_0610",
    #     "actual_res":  (28110, 18060),
    #     "actual_gsd":  5.0,
    #     "desired_gsd": 100.0,
    #     "format":      "tif",
    # },
    # "2051_102025077_D09_Arriege_E": {
    #     "image_dir":   r"Q:\02_PROJECTS\2051_102025077_D09_Arriege_E\60_UM\LVL03_CertiFLAI_0308",
    #     "actual_res":  (26460, 17004),
    #     "actual_gsd":  5.0,
    #     "desired_gsd": 100.0,
    #     "format":      "tif",
    # },
    # "2051_102025077_D09_Arriege_D": {
    #     "image_dir":   r"Q:\02_PROJECTS\2051_102025077_D09_Arriege_D\60_UM\LVL03_CertiFLAI",
    #     "actual_res":  (26460, 17004),
    #     "actual_gsd":  5.0,
    #     "desired_gsd": 100.0,
    #     "format":      "tif",
    # },
    "2025_08_10_AIR_PHKIO": {
        "image_dir":   r"Q:\00_RAWDATA\2025\8. Augustus\2025_08_10_AIR_PHKIO\Photos\2025_08_10\2020_102025032_NLIO_Misc_2025\2020_102025032_NLIO_Misc_2025\Level0\ObliqueBack",
        "actual_res":  (14204, 10652),
        "actual_gsd":  5.0,
        "desired_gsd": 100.0,
        "format":      "iiq",
    },
    "2025_08_03_AIR_FHSTG": {
        "image_dir":   r"Q:\00_RAWDATA\2025\8. Augustus\2025_08_03_AIR_FHSTG\LVL02\18148_ARIEGE_210mm_20250803a\qvRGB",
        "actual_res":  (26460, 17004),
        "actual_gsd":  5.0,
        "desired_gsd": 100.0,
        "format":      "jpg",
    },
    # "1466_102024325_Gemeente_Dronten_2025": {
    #     "image_dir":   r"Q:\02_PROJECTS\1466_102024325_Gemeente_Dronten_2025\60_UM\Lvl02-Tiff\26022025_1466_102024325\qvRGB",
    #     "actual_res":  (26460, 17004),
    #     "actual_gsd":  5.0,
    #     "desired_gsd": 100.0,
    #     "format":      "tif",
    # },
    # "1732_102024374_Gemeente_SudwestFryslan_2025s": {
    #     "image_dir":   r"Q:\02_PROJECTS\1732_102024374_Gemeente_SudwestFryslan_2025\60_UM\Lvl02-Tiff\26022025_102024374\qvRGB",
    #     "actual_res":  (26460, 17004),
    #     "actual_gsd":  5.0,
    #     "desired_gsd": 100.0,
    #     "format":      "tif",
    # },
    # "1731_102024377_Gemeente_Westerwolde_2025": {
    #     "image_dir":   r"Q:\02_PROJECTS\1731_102024377_Gemeente_Westerwolde_2025\60_UM\Lvl02-Tiff\020325_1731_102024377_west\qvRGB",
    #     "actual_res":  (26460, 17004),
    #     "actual_gsd":  5.0,
    #     "desired_gsd": 100.0,
    #     "format":      "tif",
    # }
}

TIFF_WRITE_OPTIONS: dict = {
    "driver":     "GTiff",
    "compress":   "deflate",
    "predictor":  2,
    "zlevel":     9,
    "tiled":      True,
    "blockxsize": 512,
    "blockysize": 512,
}

# Each worker holds: source array + PIL intermediate + rasterio write buffer
# Measured overhead factor is ~3x; use 3.5 for safety margin.
RAM_OVERHEAD_FACTOR = 6.0


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def compute_target_resolution(
    actual_res: tuple[int, int],
    actual_gsd: float,
    desired_gsd: float,
) -> tuple[int, int]:
    scale = actual_gsd / desired_gsd
    tw = max(1, round(actual_res[0] * scale))
    th = max(1, round(actual_res[1] * scale))
    return tw, th


def _gsd_label(gsd_cm: float) -> str:
    val = int(gsd_cm) if gsd_cm == int(gsd_cm) else gsd_cm
    return f"gsd_{val}cm"


def safe_worker_count(
    actual_res: tuple[int, int],
    dtype_bytes: int = 1,
    bands: int = 3,
    requested: int | None = None,
) -> int:
    """
    Return a worker count that won't exhaust RAM.

    Parameters
    ----------
    actual_res   : (W, H) of the full-resolution source images
    dtype_bytes  : bytes per pixel per band  (1=uint8, 2=uint16)
    bands        : number of bands
    requested    : user-supplied --workers value; None = auto
    """
    # Reserve 4GB for OS/other apps stability to prevent swapping/freezing
    reserve_bytes = 4 * 1024**3
    available_bytes = max(0, psutil.virtual_memory().available - reserve_bytes)
    
    image_bytes     = actual_res[0] * actual_res[1] * bands * dtype_bytes
    ram_per_worker  = image_bytes * RAM_OVERHEAD_FACTOR

    ram_safe  = max(1, math.floor(available_bytes / ram_per_worker))
    cpu_safe  = max(1, (os.cpu_count() or 4) - 1)
    
    # Optional: Cap max workers to 16 to avoid disk I/O bottlenecks on Q: drive
    io_cap    = 16
    
    auto      = min(ram_safe, cpu_safe, io_cap)

    image_gb = image_bytes / 1e9
    avail_gb = available_bytes / 1e9

    if requested is not None:
        chosen = min(requested, ram_safe)
        if chosen < requested:
            log.warning(
                "Requested --workers %d capped to %d to avoid OOM  "
                "(%.2f GB/image × %.1fx overhead, %.1f GB available).",
                requested, chosen, image_gb, RAM_OVERHEAD_FACTOR, avail_gb,
            )
        else:
            log.info("Workers: %d  (%.2f GB/image, %.1f GB available)", chosen, image_gb, avail_gb)
        return chosen

    log.info(
        "Auto workers: %d  (RAM safe=%d, CPU safe=%d | %.2f GB/image, %.1f GB available)",
        auto, ram_safe, cpu_safe, image_gb, avail_gb,
    )
    return auto


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def read_iiq(src_path: Path) -> tuple[np.ndarray, dict]:
    try:
        import rawpy
    except ImportError:
        raise RuntimeError("rawpy is required to read IIQ files.  pip install rawpy")

    with rawpy.imread(str(src_path)) as raw:
        # arr = raw.postprocess(
        #     use_camera_wb=True,
        #     half_size=False,
        #     no_auto_bright=True,
        #     output_color=rawpy.ColorSpace.sRGB,
        #     output_bps=16,
        # )
        
        arr = raw.postprocess(
            use_camera_wb=True,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            output_bps=8  # Change to 16 based on 16-bit output
        )


    h, w = arr.shape[:2]
    arr  = arr.transpose(2, 0, 1)   # (3, H, W)
    profile = {
        "driver": "GTiff",
        "dtype":  arr.dtype,
        "width":  w,
        "height": h,
        "count":  3,
    }
    return arr, profile


def read_standard(src_path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(src_path) as ds:
        arr     = ds.read()
        profile = ds.profile.copy()
    return arr, profile


def read_image(src_path: Path) -> tuple[np.ndarray, dict]:
    ext = src_path.suffix.lower().lstrip(".")
    if ext == "iiq":
        return read_iiq(src_path)
    return read_standard(src_path)


# ---------------------------------------------------------------------------
# Resampling  (PIL LANCZOS -- no MemoryFile, no double-buffer)
# ---------------------------------------------------------------------------

def _pil_resample(arr: np.ndarray, tgt_w: int, tgt_h: int) -> np.ndarray:
    """
    Downsample (bands, H, W) array to (tgt_w x tgt_h) with LANCZOS using PIL.

    PIL LANCZOS is equivalent in quality to GDAL's Lanczos kernel.
    Peak extra RAM = one output array (tiny vs source).
    """
    bands = arr.shape[0]

    if arr.dtype == np.uint16:
        # PIL only supports uint16 in single-band ('I;16') mode.
        # Process each band independently.
        out_bands = []
        for b in range(bands):
            pil_band = Image.fromarray(arr[b], mode="I;16")
            resized  = pil_band.resize((tgt_w, tgt_h), Image.LANCZOS)
            out_bands.append(np.array(resized, dtype=np.uint16))
        return np.stack(out_bands, axis=0)   # (bands, tgt_h, tgt_w)

    # uint8: PIL handles RGB / RGBA / L natively in one shot
    hwc  = arr.transpose(1, 2, 0)           # (H, W, bands)
    mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(bands)
    if mode:
        pil_img = Image.fromarray(hwc.squeeze(-1) if bands == 1 else hwc, mode=mode)
        out     = np.array(pil_img.resize((tgt_w, tgt_h), Image.LANCZOS), dtype=arr.dtype)
        if bands == 1:
            return out[np.newaxis, :, :]    # (1, tgt_h, tgt_w)
        return out.transpose(2, 0, 1)       # (bands, tgt_h, tgt_w)

    # Fallback: arbitrary band count, per-band
    out_bands = []
    for b in range(bands):
        pil_band = Image.fromarray(arr[b], mode="L")
        out_bands.append(np.array(pil_band.resize((tgt_w, tgt_h), Image.LANCZOS), dtype=arr.dtype))
    return np.stack(out_bands, axis=0)


def _scaled_transform(original_transform, src_w, src_h, tgt_w, tgt_h):
    if original_transform is None or original_transform == Affine.identity():
        return None
    return original_transform * Affine.scale(src_w / tgt_w, src_h / tgt_h)


def downsample_and_save(
    arr: np.ndarray,
    src_profile: dict,
    targets: list[tuple[Path, tuple[int, int]]],
) -> dict[str, bool]:
    """
    Resample *arr* to each target resolution and write lossless GeoTIFFs.

    No intermediate MemoryFile -- PIL LANCZOS operates directly on the array.
    Each target allocates only a small output array; source memory is unchanged.
    """
    results: dict[str, bool] = {}
    bands, src_h, src_w = arr.shape

    for dst_path, (tgt_w, tgt_h) in targets:
        try:
            data = _pil_resample(arr, tgt_w, tgt_h)

            out_profile = {**TIFF_WRITE_OPTIONS}
            out_profile["dtype"]  = str(data.dtype)
            out_profile["width"]  = tgt_w
            out_profile["height"] = tgt_h
            out_profile["count"]  = bands

            if src_profile.get("crs"):
                out_profile["crs"] = src_profile["crs"]

            scaled_tf = _scaled_transform(
                src_profile.get("transform"), src_w, src_h, tgt_w, tgt_h
            )
            if scaled_tf is not None:
                out_profile["transform"] = scaled_tf

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dst_path, "w", **out_profile) as dst_ds:
                dst_ds.write(data)

            results[str(dst_path)] = True

        except Exception as exc:
            logging.getLogger("aerial_downsample").error(
                "Failed to save %s: %s", dst_path, exc, exc_info=True
            )
            results[str(dst_path)] = False

    return results


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def find_images(image_dir: str | Path, fmt: str) -> list[Path]:
    root = Path(image_dir)
    if not root.exists():
        log.warning("Image directory does not exist: %s", root)
        return []
    matches = sorted(root.rglob(f"*.{fmt}")) + sorted(root.rglob(f"*.{fmt.upper()}"))
    seen, unique = set(), []
    for p in matches:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# Multiprocessing worker  (top-level -- required for spawn context pickling)
# ---------------------------------------------------------------------------

def _worker(task: dict) -> dict:
    """
    Process one image in a worker process.
    task keys: src (Path), targets (list[(Path, (W,H))]), log_level (int)
    """
    logging.basicConfig(
        level=task["log_level"],
        format="%(asctime)s [%(levelname)-8s][pid%(process)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for _lib in ("rasterio", "fiona", "rawpy", "PIL"):
        logging.getLogger(_lib).setLevel(logging.WARNING)
    wlog = logging.getLogger("aerial_downsample.worker")

    src: Path = task["src"]
    targets   = task["targets"]

    if not targets:
        return {"src": str(src), "ok": True, "skipped": True, "error": None}

    try:
        arr, profile = read_image(src)
        wlog.debug("Loaded %s  shape=%s  dtype=%s", src.name, arr.shape, arr.dtype)
    except Exception as exc:
        wlog.error("Cannot read %s: %s", src, exc)
        return {"src": str(src), "ok": False, "skipped": False, "error": str(exc)}

    results = downsample_and_save(arr, profile, targets)
    del arr   # free immediately; worker process is reused for the next task

    success = all(results.values())
    if not success:
        wlog.error("Output failures for %s: %s",
                   src.name, [p for p, ok in results.items() if not ok])

    return {
        "src":     str(src),
        "ok":      success,
        "skipped": False,
        "error":   None if success else str([p for p, ok in results.items() if not ok]),
    }


# ---------------------------------------------------------------------------
# Per-project orchestration
# ---------------------------------------------------------------------------

def process_project(
    project_name: str,
    config: dict,
    output_root: str | Path = "downsampled",
    extra_gsd: float | None = None,
    dry_run: bool = False,
    num_workers: int | None = None,   # None = auto RAM-aware
) -> dict:
    log.info("=" * 60)
    log.info("Starting project: %s", project_name)

    required = ("image_dir", "actual_res", "actual_gsd", "desired_gsd", "format")
    for k in required:
        if k not in config:
            log.error("Project '%s' missing config key '%s'", project_name, k)
            return {"project": project_name, "status": "config_error", "processed": 0, "failed": 0}

    actual_res  = config["actual_res"]
    actual_gsd  = config["actual_gsd"]
    desired_gsd = config["desired_gsd"]
    fmt         = config["format"]

    primary_res = compute_target_resolution(actual_res, actual_gsd, desired_gsd)
    log.info("Primary: %.2f -> %.2f cm/px  |  %dx%d -> %dx%d",
             actual_gsd, desired_gsd, *actual_res, *primary_res)

    extra_res: tuple[int, int] | None = None
    if extra_gsd is not None:
        extra_res = compute_target_resolution(actual_res, actual_gsd, extra_gsd)
        log.info("Extra  : %.2f -> %.2f cm/px  |  %dx%d -> %dx%d",
                 actual_gsd, extra_gsd, *actual_res, *extra_res)

    # IIQ = uint16 (2 bytes), everything else = uint8
    dtype_bytes = 2 if fmt == "iiq" else 1

    workers = safe_worker_count(
        actual_res=actual_res, dtype_bytes=dtype_bytes, bands=3, requested=num_workers
    )

    project_root = Path(output_root) / project_name
    primary_dir  = project_root / _gsd_label(desired_gsd)
    extra_dir    = (project_root / _gsd_label(extra_gsd)) if extra_gsd is not None else None

    log.info("Output (primary): %s", primary_dir)
    if extra_dir:
        log.info("Output (extra):   %s", extra_dir)

    if not dry_run:
        primary_dir.mkdir(parents=True, exist_ok=True)
        if extra_dir:
            extra_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(config["image_dir"], fmt)
    log.info("Images found: %d  (format: .%s)", len(images), fmt)
    if not images:
        log.warning("No images to process for project '%s'.", project_name)
        return {"project": project_name, "status": "no_images", "processed": 0, "failed": 0}

    # ---- build task list ----
    total_out = 1 + (1 if extra_dir is not None else 0)
    tasks: list[dict] = []
    pre_skipped = 0
    current_log_level = logging.getLogger("aerial_downsample").level or logging.INFO

    for src in images:
        primary_dst = primary_dir / f"{src.stem}.tif"
        extra_dst   = (extra_dir / f"{src.stem}.tif") if extra_dir else None

        if dry_run:
            log.info("[DRY-RUN] %s -> %s", src.name, primary_dst)
            if extra_dst:
                log.info("[DRY-RUN] %s -> %s", src.name, extra_dst)
            pre_skipped += 1
            continue

        pending: list[tuple[Path, tuple[int, int]]] = []
        done_count = 0

        if primary_dst.exists():
            done_count += 1
        else:
            pending.append((primary_dst, primary_res))

        if extra_dst is not None:
            if extra_dst.exists():
                done_count += 1
            else:
                pending.append((extra_dst, extra_res))  # type: ignore[arg-type]

        if done_count == total_out:
            pre_skipped += 1
            continue

        tasks.append({"src": src, "targets": pending, "log_level": current_log_level})

    if dry_run:
        log.info("[DRY-RUN] Would process %d images.", len(images))
        return {"project": project_name, "status": "dry_run", "processed": 0, "failed": 0}

    log.info("Tasks: %d  |  pre-skipped (already done): %d", len(tasks), pre_skipped)

    if not tasks:
        log.info("All outputs already exist, nothing to do.")
        return {
            "project": project_name, "status": "done",
            "processed": pre_skipped, "failed": 0,
            "primary_res": primary_res, "extra_res": extra_res,
        }

    # ---- execute ----
    processed = failed = 0
    t_start = time.perf_counter()

    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            result = _worker(task)
            if result["ok"]:
                processed += 1
            else:
                failed += 1
                log.error("FAILED %s: %s", result["src"], result["error"])
            _log_progress(i, len(tasks), t_start)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_worker, tasks, chunksize=1), 1
            ):
                if result["ok"]:
                    processed += 1
                else:
                    failed += 1
                    log.error("FAILED %s: %s", result["src"], result["error"])
                _log_progress(i, len(tasks), t_start)

    elapsed = time.perf_counter() - t_start
    log.info(
        "Project '%s' done -- processed: %d  failed: %d  time: %.1fs  (%.2f img/s)",
        project_name, processed, failed, elapsed,
        len(tasks) / elapsed if elapsed > 0 else 0,
    )
    return {
        "project":     project_name,
        "status":      "done",
        "processed":   processed + pre_skipped,
        "failed":      failed,
        "elapsed_s":   round(elapsed, 2),
        "primary_res": primary_res,
        "extra_res":   extra_res,
    }


def _log_progress(i: int, total: int, t_start: float, every: int = 10) -> None:
    if i % every == 0 or i == total:
        elapsed = time.perf_counter() - t_start
        rate = i / elapsed if elapsed > 0 else 0
        eta  = (total - i) / rate if rate > 0 else 0
        log.info("Progress: %d/%d  |  %.2f img/s  |  ETA %.0fs", i, total, rate, eta)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Downsample aerial imagery datasets to lossless GeoTIFFs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--project",     "-p", default="all",
                   help="Project name or 'all'. Default: all")
    p.add_argument("--output-root", "-o", default="downsampled",
                   help="Root folder for outputs. Default: ./downsampled")
    p.add_argument("--extra-gsd",   type=float, default=None, metavar="GSD_CM",
                   help="Optional second GSD (cm/pixel) to produce alongside primary.")
    p.add_argument("--dry-run",     action="store_true",
                   help="Simulate -- list what would be written, touch nothing.")
    p.add_argument("--list",        "-l", action="store_true",
                   help="List all configured projects with resolutions and exit.")
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of parallel workers. Default: auto (RAM-aware).\n"
            "The auto value is always capped by available RAM even if you\n"
            "explicitly pass --workers; a warning is logged when capping occurs."
        ),
    )
    p.add_argument("--image-dir",   help="Override image_dir for the selected project.")
    p.add_argument("--actual-gsd",  type=float, help="Override actual_gsd (cm/pixel).")
    p.add_argument("--desired-gsd", type=float, help="Override desired_gsd (cm/pixel).")
    p.add_argument("--actual-res",  nargs=2, type=int, metavar=("W", "H"),
                   help="Override actual_res: two integers W H")
    return p


def list_projects(extra_gsd: float | None = None) -> None:
    col_extra = extra_gsd is not None
    header = (
        f"\n{'Project':45s}  {'Fmt':5s}  {'Actual GSD':>10s}  "
        f"{'Primary GSD':>11s}  {'Primary res':>13s}"
    )
    if col_extra:
        header += f"  {'Extra GSD':>9s}  {'Extra res':>13s}"
    print(header)
    print("-" * (115 if col_extra else 90))
    for name, cfg in PROJECT_CONFIG.items():
        pr = compute_target_resolution(cfg["actual_res"], cfg["actual_gsd"], cfg["desired_gsd"])
        line = (
            f"{name:45s}  {cfg['format']:5s}  {cfg['actual_gsd']:>9.2f}cm"
            f"  {cfg['desired_gsd']:>10.2f}cm  {pr[0]:>6d}x{pr[1]:<6d}"
        )
        if col_extra:
            er = compute_target_resolution(cfg["actual_res"], cfg["actual_gsd"], extra_gsd)
            line += f"  {extra_gsd:>8.2f}cm  {er[0]:>6d}x{er[1]:<6d}"
        print(line)
    print()


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)
    log.setLevel(args.log_level)

    if args.list:
        list_projects(extra_gsd=args.extra_gsd)
        return 0

    if args.project.lower() == "all":
        projects = list(PROJECT_CONFIG.keys())
        log.info("Running all %d configured projects.", len(projects))
    elif args.project in PROJECT_CONFIG:
        projects = [args.project]
    else:
        log.error("Unknown project '%s'. Use --list to see available projects.", args.project)
        return 1

    if len(projects) == 1:
        name   = projects[0]
        config = dict(PROJECT_CONFIG[name])
        if args.image_dir:   config["image_dir"]   = args.image_dir
        if args.actual_gsd:  config["actual_gsd"]  = args.actual_gsd
        if args.desired_gsd: config["desired_gsd"] = args.desired_gsd
        if args.actual_res:  config["actual_res"]  = tuple(args.actual_res)
        configs = {name: config}
    else:
        configs = PROJECT_CONFIG

    if args.dry_run:
        log.info("DRY-RUN mode -- no files will be written.")
    if args.extra_gsd is not None:
        log.info("Extra GSD: %.2f cm/px (%s)", args.extra_gsd, _gsd_label(args.extra_gsd))

    summaries = []
    for project_name in projects:
        summary = process_project(
            project_name,
            configs[project_name],
            output_root=r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data\all_data",
            extra_gsd=1000,
            dry_run=args.dry_run,
            num_workers=args.workers,   # None = auto RAM-aware
        )
        summaries.append(summary)

    log.info("=" * 60)
    log.info("ALL DONE -- summary:")
    total_processed = total_failed = 0
    for s in summaries:
        tag   = "OK" if s["status"] == "done" else "!!"
        extra = (f"  extra_res={s['extra_res'][0]}x{s['extra_res'][1]}"
                 if s.get("extra_res") else "")
        log.info("  [%s] %-45s  processed=%d  failed=%d%s",
                 tag, s["project"], s.get("processed", 0), s.get("failed", 0), extra)
        total_processed += s.get("processed", 0)
        total_failed    += s.get("failed", 0)
    log.info("Total processed: %d  |  failed: %d", total_processed, total_failed)
    return 0 if total_failed == 0 else 2


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())