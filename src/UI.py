#-*-coding:utf-8 -*-
from __future__ import annotations

import argparse
import pathlib
import gradio as gr
import numpy as np
import os

os.environ['no_proxy'] = "localhost,127.0.0.1,::1"

def create_demo_hw1(process, high_contrast_process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业一: 色相/饱和度/亮度调整工具') 
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')  
                with gr.Row():
                    run_button = gr.Button(value='运行')
                with gr.Row():
                    with gr.Row():
                        saturation = gr.Slider(minimum=-1, maximum=1, step=0.01, value=0, label='饱和度')
                    with gr.Row():
                        lightness = gr.Slider(minimum=-1, maximum=1, step=0.01, value=0, label='亮度')
                    hue = gr.Slider(minimum=0, maximum=180, step=0.01, value=0, label='色相')
                with gr.Accordion('常用滤镜'):
                    black_white = gr.Button(value='黑白滤镜')
                    hight_contrast = gr.Button(value='高对比度滤镜')

            with gr.Column():
                with gr.Row():
                    output_image_hand = gr.Image(type='numpy', label='手动实现', interactive=False)
                
                with gr.Accordion('分通道显示'):
                    gallery = gr.Gallery(label='')
            with gr.Column():
                with gr.Row():
                    output_image_cv2 = gr.Image(type='numpy', label='OpenCV实现', interactive=False)
                with gr.Accordion('分通道显示'):
                    gallery_cv2 = gr.Gallery(label='')
        
        func_list = [run_button.click, hue.change, saturation.change, lightness.change, black_white.click, hight_contrast.click]
        for fn in func_list:
            fn(fn=process,
                inputs=[input_image, hue, saturation, lightness],
                outputs=[output_image_hand, gallery, output_image_cv2, gallery_cv2])
    return demo

def create_demo_hw2(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业二: 图像缩放工具') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像') 
                scale = gr.Slider(minimum=0.5, maximum=4, step=0.01, value=1, label='缩放倍数')

                transform_matrix = None
                rotate = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label='旋转角度')
                translate_x = gr.Slider(minimum=-100, maximum=100, step=1, value=0, label='垂直平移')
                translate_y = gr.Slider(minimum=-100, maximum=100, step=1, value=0, label='水平平移')
                shear_x = gr.Slider(minimum=-1, maximum=1, step=0.01, value=0, label='水平错切')
                shear_y = gr.Slider(minimum=-1, maximum=1, step=0.01, value=0, label='垂直错切')

                run_button = gr.Button(value='运行')

            with gr.Column():
                output_image = gr.Image(type='numpy', label='双线性插值-手动实现', interactive=False)
                gallery = gr.Gallery(label='手动实现')
            
            with gr.Column():
                # output_image_cv2 = gr.Image(type='numpy', label='双线性插值-手动实现', interactive=False)
                gallery_cv2 = gr.Gallery(label='OpenCV实现')

        func_list = [run_button.click, scale.change, rotate.change, translate_x.change, translate_y.change, shear_x.change, shear_y.change]
        for func in func_list:
            func(fn=process,
                inputs=[input_image, scale, rotate, translate_x, translate_y, shear_x, shear_y],
                outputs=[output_image, gallery, gallery_cv2])
    return demo


def create_demo_hw3(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业三: DCGAN图像生成工具') 
        with gr.Row():
            with gr.Column():
                input_seed = gr.Textbox(label='随机种子', value = '42')
                run_button = gr.Button(value='运行')
                output_image = gr.Image(type='numpy', label='输出图像', interactive=False)

        run_button.click(fn=process,
                        inputs=[input_seed],
                        outputs=[output_image])
    return demo

def create_demo_hw4(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业四: XXX工具') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')  
            with gr.Column():
                output_image = gr.Image(type='numpy', label='输出图像', interactive=False)
                run_button = gr.Button(value='运行')

        run_button.click(fn=process,
                        inputs=[input_image],
                        outputs=[output_image])
    return demo

def create_demo_hw5(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业五: XXX工具') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')  
            with gr.Column():
                output_image = gr.Image(type='numpy', label='输出图像', interactive=False)
                run_button = gr.Button(value='运行')

        run_button.click(fn=process,
                        inputs=[input_image],
                        outputs=[output_image])
    return demo