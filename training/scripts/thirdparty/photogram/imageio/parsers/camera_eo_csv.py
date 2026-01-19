from __future__ import annotations
import csv
import io
import re
from typing import Dict, Iterable, List, Tuple, Optional
from ...entities.camera_eo import CameraEO, OrientationAngles
from ...entities.enums import AngleConvention

MANDATORY_COLS = {"image_id", "x", "y", "z", "omega", "phi", "kappa"}
KNOWN_COLS = MANDATORY_COLS | {"crs"}  # extend here if you add optional fields


def _first_meaningful_line(lines: Iterable[str]) -> Tuple[str, List[str]]:
    """Return the first non-empty, non-comment line and a list of all lines (rewound)."""
    buf = list(lines)
    for line in buf:
        s = line.strip()
        if s and not s.startswith("#"):
            return line, buf
    raise ValueError("Input file appears empty (no data lines).")


def _detect_delimiter(header_line: str) -> Optional[str]:
    """
    Return a single-character delimiter if header contains a common delimiter.
    If none found, return None → treat as whitespace-separated.
    """
    for d in [",", ";", "\t", "|", ":"]:
        if d in header_line:
            return d
    return None  # likely whitespace-separated


def _normalize_headers(headers: List[str]) -> List[str]:
    """Lowercase + strip headers for case-insensitive matching."""
    return [h.strip().lower() for h in headers]


def _validate_headers(headers_norm: List[str]) -> None:
    missing = sorted(MANDATORY_COLS - set(headers_norm))
    if missing:
        raise ValueError(
            f"Missing mandatory columns: {', '.join(missing)}. "
            f"Found: {', '.join(headers_norm)}"
        )


def parse_camera_eo_csv(path: str) -> Dict[str, CameraEO]:
    """
    Parse camera EO text with flexible delimiters.
    Mandatory columns (case-insensitive): image_id, X, Y, Z, omega, phi, kappa
    Optional: crs

    - Supports CSV/TSV/pipe/semicolon/colon AND whitespace-separated files.
    - Ignores extra columns.
    """
    out: Dict[str, CameraEO] = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        header_line, all_lines = _first_meaningful_line(f.readlines())
        delim = _detect_delimiter(header_line)

        if delim is not None:
            # Standard csv path
            reader = csv.reader(io.StringIO("".join(all_lines)), delimiter=delim, skipinitialspace=True)
            headers = next(reader)
            headers_norm = _normalize_headers(headers)
            _validate_headers(headers_norm)

            # Build index map only for known columns
            idx = {name: i for i, name in enumerate(headers_norm) if name in KNOWN_COLS}

            for row in reader:
                if not row or (len(row) == 1 and not row[0].strip()):
                    continue  # skip blank lines
                # guard short/long rows
                if len(row) < len(headers):
                    # pad missing trailing cells
                    row = row + [""] * (len(headers) - len(row))

                # pull by index
                def val(col: str) -> str:
                    i = idx.get(col)
                    return row[i].strip() if i is not None and i < len(row) else ""

                image_id = val("image_id")
                if not image_id:
                    continue  # skip malformed line

                eo = CameraEO(
                    X=float(val("x")),
                    Y=float(val("y")),
                    Z=float(val("z")),
                    angles_deg=OrientationAngles(
                        a1=float(val("omega")),
                        a2=float(val("phi")),
                        a3=float(val("kappa")),
                        convention=AngleConvention.OPK,
                    ),
                    crs=val("crs") or None,
                )
                out[image_id] = eo

        else:
            # Whitespace-separated path (collapse multiple spaces/tabs)
            # Tokenize rows with regex on any run of whitespace
            tokens = [re.split(r"\s+", line.strip()) for line in all_lines if line.strip() and not line.lstrip().startswith("#")]
            if not tokens:
                raise ValueError("No data found after skipping blank/comment lines.")

            headers = tokens[0]
            headers_norm = _normalize_headers(headers)
            _validate_headers(headers_norm)
            idx = {name: i for i, name in enumerate(headers_norm) if name in KNOWN_COLS}

            for row in tokens[1:]:
                # pad short rows
                if len(row) < len(headers):
                    row = row + [""] * (len(headers) - len(row))

                def val(col: str) -> str:
                    i = idx.get(col)
                    return row[i].strip() if i is not None and i < len(row) else ""

                image_id = val("image_id")
                if not image_id:
                    continue

                eo = CameraEO(
                    X=float(val("x")),
                    Y=float(val("y")),
                    Z=float(val("z")),
                    angles_deg=OrientationAngles(
                        a1=float(val("omega")),
                        a2=float(val("phi")),
                        a3=float(val("kappa")),
                        convention=AngleConvention.OPK,
                    ),
                    crs=val("crs") or None,
                )
                out[image_id] = eo

    return out
