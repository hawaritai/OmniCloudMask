from pathlib import Path

# paths
input_dir = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data\2051_102025082_D82_Tarn-et-Garonne_A\images")
names_txt = Path(r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\scripts\test\tmp.txt")

# read allowed names
with open(names_txt, "r") as f:
    allowed = {line.strip() for line in f if line.strip()}

# dry run first
DRY_RUN = False  # set to False to actually delete

removed = []
kept = []

for tif in input_dir.glob("*.tif"):
    name_no_ext = tif.stem

    if name_no_ext not in allowed:
        removed.append(tif.name)
        if not DRY_RUN:
            tif.unlink()
    else:
        kept.append(tif.name)

# report
print(f"Kept: {len(kept)}")
print(f"Removed: {len(removed)}")

if removed:
    print("\nFiles to be removed:")
    for f in removed:
        print(" ", f)