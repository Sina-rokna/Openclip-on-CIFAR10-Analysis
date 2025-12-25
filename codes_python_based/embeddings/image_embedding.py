import torch
import numpy as np
from tqdm import tqdm 

def compute_image_embedding(model, loader, device):
    feats, labels = [], []
    with torch.no_grad():
        for imgs, y in tqdm(loader):
            imgs = imgs.to(device)
            f = model.encode_image(imgs).cpu().numpy()
            feats.append(f)
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)
