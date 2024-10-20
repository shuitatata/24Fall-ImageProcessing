#-*-coding:utf-8 -*-
from __future__ import annotations

import argparse
import pathlib
import gradio as gr
import numpy as np


def create_demo_hw1(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业一: XXX工具') 
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