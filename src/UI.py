#-*-coding:utf-8 -*-
from __future__ import annotations

import argparse
import pathlib
import gradio as gr
import numpy as np
import os

os.environ['no_proxy'] = "localhost,127.0.0.1,::1"

def create_demo_hw1(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业一: XXX工具') 
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
                
            with gr.Column():
                with gr.Row():
                    output_image_hand = gr.Image(type='numpy', label='手动实现', interactive=False)
                
                with gr.Accordion('分通道显示'):
                    gallery = gr.Gallery(label='')
                    # output_image_h = gr.Image(type='numpy', label='色相', interactive=False)
                    # output_image_l = gr.Image(type='numpy', label='亮度', interactive=False)
                    # output_image_s = gr.Image(type='numpy', label='饱和度', interactive=False)
            with gr.Column():
                with gr.Row():
                    output_image_cv2 = gr.Image(type='numpy', label='OpenCV实现', interactive=False)
                with gr.Accordion('分通道显示'):
                    gallery_cv2 = gr.Gallery(label='')
                    # output_image_h_cv2 = gr.Image(type='numpy', label='色相', interactive=False)
                    # output_image_l_cv2 = gr.Image(type='numpy', label='亮度', interactive=False)
                    # output_image_s_cv2 = gr.Image(type='numpy', label='饱和度', interactive=False)
                    
        run_button.click(fn=process,
                        inputs=[input_image, hue, saturation, lightness],
                        outputs=[output_image_hand, gallery, output_image_cv2, gallery_cv2])
        hue.change(fn=process,
                    inputs=[input_image, hue, saturation, lightness],
                    outputs=[output_image_hand, gallery, output_image_cv2, gallery_cv2])
        saturation.change(fn=process,
                    inputs=[input_image, hue, saturation, lightness],
                    outputs=[output_image_hand, gallery, output_image_cv2, gallery_cv2])
        lightness.change(fn=process,
                    inputs=[input_image, hue, saturation, lightness],
                    outputs=[output_image_hand, gallery, output_image_cv2, gallery_cv2])
    return demo

def create_demo_hw2(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业二: XXX工具') 
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


def create_demo_hw3(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业三: XXX工具') 
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