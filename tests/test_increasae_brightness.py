import cv2
import numpy as np
from pathlib import Path

input_dir = Path(r"\\k10dtnqumulo\HIT\temp\training\data\2051_102025077_D09_Arriege_E\GT_d")
output_dir = Path(r"\\k10dtnqumulo\HIT\temp\training\data\2051_102025077_D09_Arriege_E\2")
output_dir.mkdir(parents=True, exist_ok=True)

ALPHA = 1.0   # contrast
BETA = 20    # brightness

for tif_path in input_dir.glob("*.tif"):
    img = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"❌ Failed to read: {tif_path}")
        continue

    bright = cv2.convertScaleAbs(img, alpha=ALPHA, beta=BETA)

    out_path = output_dir / tif_path.name
    ok = cv2.imwrite(str(out_path), bright)

    if not ok:
        print(f"❌ Failed to write: {out_path}")
    else:
        print(f"✅ Saved: {tif_path.name}")
