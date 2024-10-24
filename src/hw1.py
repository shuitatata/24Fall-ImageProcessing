import numpy as np
import cv2

def rgb2hls(input_image):
    input_image = input_image.astype(np.float32) / 255.0
    B, G, R = input_image[:,:,0], input_image[:,:,1], input_image[:,:,2]

    max_val = np.max(input_image, axis=2)
    min_val = np.min(input_image, axis=2)

    L = (max_val + min_val) / 2.0

    S = np.zeros_like(L)
    mask = L < 0.5
    S[mask] = (max_val[mask] - min_val[mask]) / (max_val[mask] + min_val[mask] + 1e-10)
    mask = L >= 0.5
    S[mask] = (max_val[mask] - min_val[mask]) / (2.0 - max_val[mask] - min_val[mask] + 1e-10)

    H = np.zeros_like(L)
    
    H = np.where(max_val - min_val == 0, 0, H)

    mask = np.logical_and(R == max_val, max_val != min_val)
    H[mask] = (G[mask] - B[mask]) / (max_val[mask] - min_val[mask])
    mask = np.logical_and(G == max_val, max_val != min_val)
    H[mask] = 2.0 + (B[mask] - R[mask]) / (max_val[mask] - min_val[mask])
    mask = np.logical_and(B == max_val, max_val != min_val)
    H[mask] = 4.0 + (R[mask] - G[mask]) / (max_val[mask] - min_val[mask])

    H = H / 6.0 % 1.0
    H = (H * 180).astype(np.uint32)

    output_image = np.stack([H, (L * 255).astype(np.uint8), (S * 255).astype(np.uint8)], axis=2)

    return output_image
