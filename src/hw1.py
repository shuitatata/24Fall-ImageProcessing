import numpy as np
import cv2

def rgb2hls(input_image):
    input_image = input_image.astype(np.float32) / 255.0
    B, G, R = input_image[:,:,0], input_image[:,:,1], input_image[:,:,2]

    min_val = np.min(input_image, axis=2)
    max_val = np.max(input_image, axis=2)

    with np.errstate(divide='ignore', invalid='ignore'):
        S = 1.0 - 3 * min_val / (R + G + B)
    S = np.where(R + G + B == 0, 0.0, S)

    L = (R + G + B) / 3.0

    R *= 255
    G *= 255
    B *= 255
    numerator = 2*R - G - B
    denominator = 2 * np.sqrt(R**2 + G**2 + B**2 - R*G - R*B - G*B)
    R /= 255.0
    G /= 255.0
    B /= 255.0

    with np.errstate(divide='ignore', invalid='ignore'):
        theta = np.arccos(numerator / denominator)
    
    # 原为 theta = np.where(denominator == 0, np.where(numerator > 0, 1, -1), theta)
    theta = np.where(denominator == 0, 0.0, theta)

    H = np.where(B <= G, theta, 2*np.pi - theta)

    H = H / (2*np.pi) * 180

    L = np.clip(L * 255, 0, 255)
    S = np.clip(S * 255, 0, 255)
    return np.stack([H, L, S], axis=-1).astype(np.uint8)

def hls2bgr(input_image):
    # 提取 H, L, S 通道，H 的范围是 [0, 180]，L 和 S 的范围是 [0, 255]
    input_image = input_image.astype(np.float32)
    H, L, S = input_image[:,:,0], input_image[:,:,1], input_image[:,:,2]
    L = L / 255.0
    S = S / 255.0
    
    # 将 H 从角度制 [0, 180] 转换为弧度制 [0, π]
    H = H / 180.0 * 2.0 * np.pi
    
    # 初始化 R, G, B 通道
    R = np.zeros_like(H, dtype=np.float32)
    G = np.zeros_like(H, dtype=np.float32)
    B = np.zeros_like(H, dtype=np.float32)

    # 第一个色相区间 0 <= H < 2π/3
    mask1 = np.logical_and(H >= 0, H < 2.0 * np.pi / 3.0)
    H1 = H[mask1]
    R[mask1] = L[mask1] * (1.0 + (S[mask1] * np.cos(H1)) / np.cos(np.pi / 3.0 - H1))
    B[mask1] = L[mask1] * (1.0 - S[mask1])
    G[mask1] = 3.0 * L[mask1] - (R[mask1] + B[mask1])

    # 第二个色相区间 2π/3 <= H < 4π/3
    mask2 = np.logical_and(H >= 2.0 * np.pi / 3.0, H < 4.0* np.pi / 3.0)
    H2 = H[mask2] - 2.0 * np.pi / 3.0  # 色相偏移
    G[mask2] = L[mask2] * (1.0 + (S[mask2] * np.cos(H2)) / np.cos(np.pi / 3.0 - H2))
    R[mask2] = L[mask2] * (1.0 - S[mask2])
    B[mask2] = 3.0 * L[mask2] - (R[mask2] + G[mask2])

    # 第三个色相区间 4π/3 <= H < 2π
    mask3 = np.logical_and(H >= 4.0 * np.pi / 3.0, H < 2.0 * np.pi)
    H3 = H[mask3] - 4.0 * np.pi / 3.0  # 色相偏移
    B[mask3] = L[mask3] * (1.0 + (S[mask3] * np.cos(H3)) / np.cos(np.pi / 3.0 - H3))
    G[mask3] = L[mask3] * (1.0 - S[mask3])
    R[mask3] = 3.0 * L[mask3] - (G[mask3] + B[mask3])

    # 将 R, G, B 通道合并成 RGB 图像

    B = np.clip(B * 255, 0, 255)
    G = np.clip(G * 255, 0, 255)
    R = np.clip(R * 255, 0, 255)

    rgb_image = np.stack([B, G, R], axis=-1)
    
    # 将 RGB 值限制在 [0, 255] 之间，并转换为 uint8 类型
    rgb_image = rgb_image.astype(np.uint8)
    
    return rgb_image

    
