#-*-coding:UTF-8 -*-
import numpy as np
import gradio as gr
import cv2
import hw1

def function_hw1(input_image, hue, saturation, lightness):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)
    output_image_cv2 = input_image.copy()
    
    # print(input_image.dtype)

    hls_image = hw1.rgb2hls(input_image)
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


def function_hw2(input_image):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)    
    output_image = input_image
    # 请补充作业2的图像处理代码
    return output_image

def function_hw3(input_image):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)   
    output_image = input_image
    # 请补充作业3的图像处理代码
    return output_image

def function_hw4(input_image):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)
    output_image = input_image
    # 请补充作业4的图像处理代码
    return output_image

def function_hw5(input_image):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)
    output_image = input_image
    # 请补充作业5的图像处理代码
    return output_image