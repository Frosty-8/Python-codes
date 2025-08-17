from scipy.linalg import sqrtm
import numpy as np
from rich import print as rprint

def calculate_fid(u1, sigma1, u2, sigma2):
    diff = u1 - u2
    covmean = sqrtm(sigma1.dot(sigma2))
    fid = diff.dot(diff) - np.trace(sigma1 + sigma2 - 2*covmean)
    return np.real(fid)

u1, sigma1 = np.random.rand(3), np.eye(3)
u2, sigma2 = np.random.rand(3), np.eye(3)

rprint(f"FID SCORE : {calculate_fid(u1, sigma1, u2, sigma2)}")