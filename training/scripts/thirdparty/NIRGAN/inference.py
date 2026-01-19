from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset, DataLoader
import os
from skimage.exposure import match_histograms
import torch.nn.functional as F 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# GPU Settings --------
# set visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"



# Load model
model = Px2Px_PL(config)
model.load_state_dict(torch.load("logs/best/S2.ckpt")['state_dict'], strict=False)
model = model.eval().to(device)

# Run inference
for batch in dataloader:
    hr_rgb = batch["hr"].to(device)
    with torch.no_grad():
        synth_nir = model(hr_rgb).cpu()
    
    # Optional: Match predicted NIR to Sentinel-2 NIR histogram
    matched_nir = histogram_match(synth_nir, batch["s2_nir"])
    
    # Save NIR bands
    for nir, name in zip(matched_nir, batch["id"]):
        save_image(nir, "data/synthDataset/synth_nirs", name)