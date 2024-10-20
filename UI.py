#-*-coding:GBK -*-
from __future__ import annotations

import argparse
import pathlib
import gradio as gr
import numpy as np


def create_demo_hw1(process):
    with gr.Blocks() as demo:
        gr.Markdown('## ��ҵһ: XXX����') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='����ͼ��')  
            with gr.Column():
                output_image = gr.Image(type='numpy', label='���ͼ��', interactive=False)
                run_button = gr.Button(value='����')

        run_button.click(fn=process,
                        inputs=[input_image],
                        outputs=[output_image])
    return demo

def create_demo_hw2(process):
    with gr.Blocks() as demo:
        gr.Markdown('## ��ҵ��: XXX����') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='����ͼ��')  
            with gr.Column():
                output_image = gr.Image(type='numpy', label='���ͼ��', interactive=False)
                run_button = gr.Button(value='����')

        run_button.click(fn=process,
                        inputs=[input_image],
                        outputs=[output_image])
    return demo


def create_demo_hw3(process):
    with gr.Blocks() as demo:
        gr.Markdown('## ��ҵ��: XXX����') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='����ͼ��')  
            with gr.Column():
                output_image = gr.Image(type='numpy', label='���ͼ��', interactive=False)
                run_button = gr.Button(value='����')

        run_button.click(fn=process,
                        inputs=[input_image],
                        outputs=[output_image])
    return demo

def create_demo_hw4(process):
    with gr.Blocks() as demo:
        gr.Markdown('## ��ҵ��: XXX����') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='����ͼ��')  
            with gr.Column():
                output_image = gr.Image(type='numpy', label='���ͼ��', interactive=False)
                run_button = gr.Button(value='����')

        run_button.click(fn=process,
                        inputs=[input_image],
                        outputs=[output_image])
    return demo

def create_demo_hw5(process):
    with gr.Blocks() as demo:
        gr.Markdown('## ��ҵ��: XXX����') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='����ͼ��')  
            with gr.Column():
                output_image = gr.Image(type='numpy', label='���ͼ��', interactive=False)
                run_button = gr.Button(value='����')

        run_button.click(fn=process,
                        inputs=[input_image],
                        outputs=[output_image])
    return demo