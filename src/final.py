import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util import util
from torchvision import transforms
from PIL import Image
import torch
import cv2
from openai import OpenAI
import base64
import re

client = OpenAI(
    api_key = os.environ['OPENAI_API_KEY'],
    timeout=20,
    base_url="https://api.feidaapi.com/v1"
)

proxies = {
    "http": 'http://127.0.0.1:7890',
    "https": 'http://127.0.0.1:7890'
}

def create_image(input_image):
    opt = TestOptions().parse_demo()

    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.no_dropout = True

    model_path = 'latest_net'
    model = create_model(opt)
    model.setup_demo(opt, model_path)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    input_image = transform(input_image).type(torch.FloatTensor)
    input_image = input_image.unsqueeze(0)

    data = {'A': input_image, 'A_paths': 'demo'}

    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()

    for label, image in visuals.items():
        if label == 'real':
            continue
        image_result = util.tensor2im(image)
        return image_result

def encode_image(image):
    return base64.b64encode(cv2.imencode('.jpg', image)[1]).decode("utf-8")

def generate_poem(image):
    base64_image = encode_image(image)
    # response = client.chat.completions.create(
    #     model='gpt-4o-mini',
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     'type': 'text',
    #                     'text': '我现在要发一个朋友圈，需要撰写一个文案。请你先阅读并描述这个黑白水墨图片的内容，随后结合画面意境引用一两句有关的中国古典诗词。注意，图片是黑白水墨风格的风景，白色不一定是雪'
    #                 },
    #                 {
    #                     'type': 'image_url',
    #                     'image_url': {"url": f"data:image/jpeg;base64,{base64_image}"}
    #                 }
    #             ]
    #         }
    #     ],
    # )

    # content = response.choices[0].message.content

    # print(content)

    # poem_lines = re.findall(r'“([^”]+)”', content)
    # for line in poem_lines:
    #     print(line)

    return "青山不墨千秋画，绿水无弦万古琴。"


if __name__ == '__main__':
    img = create_image(Image.open('test.jpg'))

    print("generate done!")

    print(generate_poem(img))

    cv2.imshow('result', img)
    cv2.waitKey(0)

