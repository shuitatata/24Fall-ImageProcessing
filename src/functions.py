#-*-coding:UTF-8 -*-
import numpy as np
import gradio as gr
import cv2
import hw1
import hw2
from hw3 import Generator
import hw4
import hw5
import torch
import torchvision.utils as vutils

def function_hw1(input_image, hue, saturation, lightness):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)
    output_image_cv2 = input_image.copy()
    # print(input_image.dtype)

    hls_image = hw1.bgr2hls(input_image)
    hls_image = hls_image.astype(np.float32)
    hls_image[:,:,0] = (hls_image[:,:,0] + hue) % 180
    hls_image[:,:,1] = np.clip(hls_image[:,:,1] * (1+lightness), 0, 255)
    hls_image[:,:,2] = np.clip(hls_image[:,:,2] * (1+saturation), 0, 255)
    hls_image = hls_image.astype(np.uint8)
    
    hls_image_cv2 = cv2.cvtColor(output_image_cv2, cv2.COLOR_BGR2HLS)
    hls_image_cv2 = hls_image_cv2.astype(np.float32)
    hls_image_cv2[:,:,0] = (hls_image_cv2[:,:,0] + hue) % 180
    hls_image_cv2[:,:,1] = np.clip(hls_image_cv2[:,:,1] * (1+lightness), 0, 255)
    hls_image_cv2[:,:,2] = np.clip(hls_image_cv2[:,:,2] * (1+saturation), 0, 255)
    hls_image_cv2 = hls_image_cv2.astype(np.uint8)
    output_image_cv2 = cv2.cvtColor(hls_image_cv2, cv2.COLOR_HLS2BGR)
    hls_image_list_cv2 = [(hls_image_cv2[:,:,0], '色相'), (hls_image_cv2[:,:,1], '亮度'), (hls_image_cv2[:,:,2], '饱和度'), 
        (output_image_cv2[:,:,0], 'B'), (output_image_cv2[:,:,1], 'G'), (output_image_cv2[:,:,2], 'R')]

    output_image = hw1.hls2bgr(hls_image)
    # output_image = cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR)
    hls_image_list = [(hls_image[:,:,0], '色相'), (hls_image[:,:,1], '亮度'), (hls_image[:,:,2], '饱和度'), 
        (output_image[:,:,0], 'B'), (output_image[:,:,1], 'G'), (output_image[:,:,2], 'R')]
    return output_image, hls_image_list, output_image_cv2, hls_image_list_cv2

def high_contrast_hw1(input_image):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)
    output_image = input_image.copy()

    hls_image = hw1.bgr2hls(input_image)
    hls_image = hls_image.astype(np.float32)

    mask = hls_image[:,:,1] < 128
    hls_image[mask, 1] = hls_image[mask, 1] * 0.7
    hls_image[~mask, 1] = np.clip(hls_image[~mask, 1] * 1.3, 0, 255)

    hls_image = hls_image.astype(np.uint8)
    output_image = hw1.hls2bgr(hls_image)
    hls_image_list = [(hls_image[:,:,0], '色相'), (hls_image[:,:,1], '亮度'), (hls_image[:,:,2], '饱和度'),
        (output_image[:,:,0], 'B'), (output_image[:,:,1], 'G'), (output_image[:,:,2], 'R')]

    hls_image_cv2 = cv2.cvtColor(input_image, cv2.COLOR_BGR2HLS)
    hls_image_cv2 = hls_image_cv2.astype(np.float32)

    mask = hls_image_cv2[:,:,1] < 128
    hls_image_cv2[mask, 1] = hls_image_cv2[mask, 1] * 0.7
    hls_image_cv2[~mask, 1] = np.clip(hls_image_cv2[~mask, 1] * 1.3, 0, 255)

    hls_image_cv2 = hls_image_cv2.astype(np.uint8)
    output_image_cv2 = cv2.cvtColor(hls_image_cv2, cv2.COLOR_HLS2BGR)
    hls_image_list_cv2 = [(hls_image_cv2[:,:,0], '色相'), (hls_image_cv2[:,:,1], '亮度'), (hls_image_cv2[:,:,2], '饱和度'),
        (output_image_cv2[:,:,0], 'B'), (output_image_cv2[:,:,1], 'G'), (output_image_cv2[:,:,2], 'R')]
    return output_image, hls_image_list, output_image_cv2, hls_image_list_cv2

def function_hw2(input_image, scale, rotate, translate_x, translate_y, shear_x, shear_y):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)    
    output_image = input_image
    
    nearest_cv2 = cv2.resize(output_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    bilinear_cv2 = cv2.resize(output_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    bicubic_cv2 = cv2.resize(output_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    nearest = hw2.resize_nearest(output_image, scale)
    bilinear = hw2.resize_bilinear(output_image, scale)
    bicubic = hw2.resize_bicubic(output_image, scale)

    transform_matrix = hw2.get_rotation_matrix(rotate)
    transform_matrix = hw2.get_translation_matrix(translate_x/100.0* output_image.shape[1], translate_y/100.0* output_image.shape[0]) @ transform_matrix
    transform_matrix = hw2.get_shear_matrix(shear_x, shear_y) @ transform_matrix

    print("processing transform_matrix")
    output = hw2.apply_transform(bilinear, transform_matrix, bilinear.shape[:2])
    print("done1")

    # output_cv2 = cv2.warpAffine(bilinear_cv2, transform_matrix[:2], bilinear_cv2.shape[:2][::-1], flags=cv2.INTER_LINEAR)

    output_images = [(nearest, '最近邻插值'), (bilinear, '双线性插值'), (bicubic, '双三次插值')]
    output_images_cv2 = [(nearest_cv2, '最近邻插值'), (bilinear_cv2, '双线性插值'), (bicubic_cv2, '双三次插值')]

    return output, output_images, output_images_cv2

def function_hw3(seed = 2024):
    '''
    根据输入的种子生成一张图片
    '''
    
    # print(seed)

    try:
        seed = int(seed)
    except:
        raise gr.Error('输入错误：种子必须为整数', duration=5)

    # set the seed
    torch.manual_seed(seed)
    # create the generator
    netG = Generator(0).to("cpu")
    # load the weights
    netG.load_state_dict(torch.load("./checkpoint/netG.pth", map_location="cpu", weights_only=False))
    # generate the image
    with torch.no_grad():
        image = netG(torch.randn(1, 100, 1, 1))
    # convert to numpy
    image = image.cpu().detach().numpy() # [1, 3, 64, 64]

    # Convert the generated image to a format suitable for display with OpenCV
    image = (image.squeeze().transpose(1, 2, 0) * 127.5 + 127.5).astype(np.uint8)

    # fake_image = cv2.cvtColor(fake_image, cv2.COLOR_RGB2BGR)
    
    # cv2.imshow("fake", cv2.resize(fake, (256, 256)))
    # cv2.waitKey(0)
    return image
    # return output_image

def function_hw4(input_image, guided_image=None, r=4, eps=0.01, sigma_color=75, sigma_space=75, lambda_factor=1, mode='导向滤波'):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)
    # output_image = input_image
    if guided_image is None:
        guided_image = input_image
    
    if mode == '导向滤波':
        sharpen_image = input_image.copy()

        input_image = input_image.astype(np.float32) / 255.0
        guided_image = guided_image.astype(np.float32) / 255.0
        
        filtered_image = hw4.guided_filter(guided_image, input_image, r, eps)
        filtered_image = (filtered_image * 255).astype(np.uint8)

        sharpened_image = hw4.sharpen_guided(sharpen_image, r, eps, lambda_factor)
        # sharpened_image = (sharpened_image * 255).astype(np.uint8)
        return filtered_image, sharpened_image
    
    elif mode == '双边滤波':
        sharpen_image = input_image.copy()

        input_image = input_image.astype(np.float32) / 255.0
        filtered_image = hw4.bilateral_filter(input_image, r, sigma_color, sigma_space)
        filtered_image = (filtered_image * 255).astype(np.uint8)

        sharpened_image = hw4.sharpen_bilateral(sharpen_image, r, sigma_color, sigma_space, lambda_factor)
        # sharpened_image = (sharpened_image * 255).astype(np.uint8)
        return filtered_image, sharpened_image
    else :
        raise gr.Error('输入错误：请选择正确的滤波模式', duration=5)

def function_hw5(input_image):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)
    output_image = input_image
    # 请补充作业5的图像处理代码

    output_image_1 = hw5.histogram_equalization(output_image)
    output_image_2 = hw5.clahe_algorithm(output_image)
    
    return output_image_1, output_image_2

if __name__ == '__main__':
    function_hw3(999)