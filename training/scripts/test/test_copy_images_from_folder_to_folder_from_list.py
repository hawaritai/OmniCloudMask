from pathlib import Path
import shutil

# paths
input_dir = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data\all_data\2025_08_10_AIR_PHKIO\gsd_100cm")
names_txt = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\scripts\test\tmp.txt")
output_dir = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data\all_images_v3")

output_dir.mkdir(parents=True, exist_ok=True)

# read names from txt (strip spaces / empty lines)
with open(names_txt, "r") as f:
    names = [line.strip() for line in f if line.strip()]

# copy files
missing = []
for name in names:
    src = input_dir / f"{name}.tif"
    dst = output_dir / src.name

    if src.exists():
        shutil.copy2(src, dst)
    else:
        missing.append(src.name)

# report
print(f"Copied {len(names) - len(missing)} files")
if missing:
    print("Missing files:")
    for m in missing:
        print("  ", m)
