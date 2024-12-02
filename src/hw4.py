import cv2
import numpy as np
import pandas as pd

def guided_filter(I, q, r, eps):
    """
    实现导向滤波
    Args:
        I: 导向图像 
        q: 输入图像
        r: 滤波窗口半径
        eps: 正则化参数
    
    Returns:
        过滤后的图像
    """
    # 计算窗口内均值
    kernel = (2 * r + 1, 2 * r + 1)
    mean_I = cv2.blur(I, kernel)  # I 的均值
    mean_q = cv2.blur(q, kernel)  # q 的均值
    mean_Iq = cv2.blur(I * q, kernel)  # I 和 q 的联合均值
    
    # 计算协方差和方差
    cov_Iq = mean_Iq - mean_I * mean_q  # 协方差
    mean_II = cv2.blur(I * I, kernel)  # I 的平方均值
    var_I = mean_II - mean_I * mean_I  # I 的方差
    
    # 计算线性系数 a 和 b
    a = cov_Iq / (var_I + eps)
    b = mean_q - a * mean_I
    
    # 计算 a 和 b 的均值
    mean_a = cv2.blur(a, kernel)
    mean_b = cv2.blur(b, kernel)
    
    # 输出图像
    q_out = mean_a * I + mean_b
    return q_out

def sharpen_guided(img, r, eps, lambda_factor):
    # 如果是彩色图像，转换为灰度图作为导向图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    else:
        gray = img / 255.0

    # 对每个通道分别处理
    if len(img.shape) == 3:  # 彩色图像
        img = img / 255.0
        sharpened_img = np.zeros_like(img)
        for i in range(3):  # 对 R、G、B 通道分别处理
            smooth = guided_filter(gray, img[:, :, i], r, eps)
            high_freq = img[:, :, i] - smooth
            sharpened_img[:, :, i] = img[:, :, i] + lambda_factor * high_freq
    else:  # 灰度图像
        img = img / 255.0
        smooth = guided_filter(gray, img, r, eps)
        high_freq = img - smooth
        sharpened_img = img + lambda_factor * high_freq

    # 将图像恢复到 8 位范围
    sharpened_img = np.clip(sharpened_img, 0, 1) * 255
    return sharpened_img.astype(np.uint8)

def bilateral_filter(img, d, sigma_color, sigma_space):
    """
    实现双边滤波
    Args:
        img: 输入图像
        d: 滤波窗口直径
        sigma_color: 颜色空间滤波器的 sigma 值
        sigma_space: 坐标空间滤波器的 sigma 值
    
    Returns:
        过滤后的图像
    """
    half_d = d // 2
    img_padded = np.pad(img, ((half_d, half_d), (half_d, half_d), (0, 0)), mode='reflect')
    filtered_img = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                i_min = i
                i_max = i + d
                j_min = j
                j_max = j + d

                region = img_padded[i_min:i_max, j_min:j_max, k]
                center_pixel = img_padded[i + half_d, j + half_d, k]

                spatial_weights = np.exp(-((np.arange(d) - half_d) ** 2 + (np.arange(d)[:, None] - half_d) ** 2) / (2 * sigma_space ** 2))
                color_weights = np.exp(-(region - center_pixel) ** 2 / (2 * sigma_color ** 2))

                weights = spatial_weights * color_weights
                weights /= np.sum(weights)

                filtered_img[i, j, k] = np.sum(weights * region)

    return filtered_img

def sharpen_bilateral(img, d, sigma_color, sigma_space, lambda_factor):
    img = img.astype(np.float32) / 255

    sharpened_img = np.zeros_like(img)

    smooth = bilateral_filter(img, d, sigma_color, sigma_space)
    high_freq = img - smooth
    sharpened_img = img + lambda_factor * high_freq

    sharpened_img = np.clip(sharpened_img, 0, 1) * 255
    return sharpened_img.astype(np.uint8)

if __name__ == '__main__':
    # Read image
    img = cv2.imread('./pic/test_image/cat512.jpg')
    # img = img.astype(np.float32) / 255
    
    # Add noise
    # noise = np.random.randn(*img.shape) * 0.1
    # noisy_img = img + noise
    # gray_img = cv2.cvtColor(noisy_img.astype(np.float32), cv2.COLOR_BGR2GRAY)

    # Bilateral filter
    r = 4
    sigma_color = 75
    sigma_space = 75
    lambda_factor = 3
    # bilateral_img = bilateral_filter(noisy_img, r, sigma_color, sigma_space)
    # bilateral_img = np.clip(bilateral_img, 0, 1)
    # bilateral_img = (bilateral_img * 255).astype(np.uint8)
    sharpen_img_b = sharpen_bilateral(img, r, sigma_color, sigma_space, lambda_factor)
    # sharpen_img_b = np.clip(sharpen_img_b, 0, 1)
    # sharpen_img_b = (sharpen_img_b * 255).astype(np.uint8)

    # Guided filter
    r, eps = 4, 0.02
    # guided_img = np.zeros_like(img)

    # for i in range(3):
    #     guided_img[:, :, i] = guided_filter(gray_img, noisy_img[:, :, i], r, eps)
    # guided_img = np.clip(guided_img, 0, 1)
    # guided_img = (guided_img * 255).astype(np.uint8)
    # filtered_img = guided_filter(gray_img, noisy_img, r, eps)

    # Sharpening
    lambda_factor = 3
    sharpened_img_f = np.zeros_like(img)
    for i in range(3):
        sharpened_img_f[:, :, i] = sharpen_guided(img[:, :, i], r, eps, lambda_factor)
    # sharpen_img_f = np.clip(sharpened_img_f, 0, 1)
    # sharpen_img_f = (sharpen_img_f * 255).astype(np.uint8)

    cv2.imwrite('./pic/hw4/cat2_256_sharpen_guided.jpg', sharpened_img_f)
    # noisy_img = np.clip(noisy_img, 0, 1)
    # noisy_img = (noisy_img * 255).astype(np.uint8)
    # cv2.imwrite('./pic/hw4/cat2_256_guided.jpg', guided_img)
    # cv2.imwrite('./pic/hw4/cat2_256_noisy_guided.jpg', noisy_img)
    # cv2.imshow('Guided Filtered Image', noisy_img)
    # cv2.waitKey(0)


    cv2.imwrite('./pic/hw4/cat2_256_sharpen_bilateral.jpg', sharpen_img_b)
    # cv2.imwrite('./pic/hw4/cat2_256_bilateral.jpg', bilateral_img)
    # cv2.imwrite('./pic/hw4/cat2_256_noisy_bilateral.jpg', noisy_img)

    # cv2.imshow('noisy Image', noisy_img)
    # cv2.imshow('Bilateral Filtered Image', bilateral_img)
    # cv2.waitKey(0)

    
    

    


    # cv2.imshow('Original Image', img)
    # cv2.imshow('Noisy Image', noisy_img)
    # cv2.imshow('Filtered Image', filtered_img)
    # cv2.waitKey(0)
