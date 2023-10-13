import argparse
import base64
import time
from io import BytesIO

import numpy as np
import requests
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *


# helper functions to encode and decode images
def encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue())
    return img_str


def decode_image(img):
    buff = BytesIO(base64.b64decode(img.encode("utf8")))
    image = Image.open(buff)
    return image


def main(model_name):
    # client = httpclient.InferenceServerClient(url="localhost:8000")
    client = httpclient.InferenceServerClient(url="localhost:5555")

    # Inputs
    # image = Image.open("./test_images/2_trees.jpg")
    # mask = Image.open("test_images/2_trees.png")
    image = Image.open("./test_images/cat.jpg")
    mask = Image.open("test_images/cat.png")
    prompt = "robot"

    image = encode_image(image).decode("utf8")
    mask = encode_image(mask).decode("utf8")

    image = np.asarray([image], dtype=object)
    mask = np.asarray([mask], dtype=object)
    prompt = np.asarray([prompt], dtype=object)

    # Set Inputs
    input_tensors = [
        httpclient.InferInput("image", [1], datatype="BYTES"),
        httpclient.InferInput("mask", [1], datatype="BYTES"),
        httpclient.InferInput("prompt", [1], datatype="BYTES"),
    ]

    input_tensors[0].set_data_from_numpy(image.reshape([1]))
    input_tensors[1].set_data_from_numpy(mask.reshape([1]))
    input_tensors[2].set_data_from_numpy(prompt.reshape([1]))

    # Set outputs
    outputs = [httpclient.InferRequestedOutput("generated_image")]

    # Query
    t1 = time.perf_counter()
    query_response = client.infer(
        model_name=model_name, inputs=input_tensors, outputs=outputs
    )

    print(time.perf_counter() - t1)
    
    # Output
    generated_image = query_response.as_numpy("generated_image")

    # print(type(generated_image), generated_image[0])
    decoded_images = []
    
    encoded_image = generated_image[0]
    
    image_data = base64.b64decode(encoded_image)

    # Создайте объект BytesIO для считывания байтов
    image_buffer = BytesIO(image_data)

    # Откройте изображение с использованием PIL
    img = Image.open(image_buffer)
    img.save("./test_images/output.jpeg")



    # Добавьте изображение в список раскодированных изображений
    # decoded_images.append(img)
    

    # for encoded_image in generated_image:
    #     # Декодируйте строку base64 обратно в байты
    #     image_data = base64.b64decode(encoded_image)

    #     # Создайте объект BytesIO для считывания байтов
    #     image_buffer = BytesIO(image_data)

    #     # Откройте изображение с использованием PIL
    #     img = Image.open(image_buffer)
    #     img.save("./test_images/output.jpeg")

    #     # Добавьте изображение в список раскодированных изображений
    #     decoded_images.append(img)

    # img = Image.frombuffer(generated_image[0])

    # images = [decode_image(i) for i in generated_image]
    # images = decode_image(generated_image[0])
    # if generated_image.ndim == 3:
    #     generated_image = generated_image[None, ...]

    # generated_image = (generated_image * 255).round().astype("uint8")
    # pil_images = [Image.fromarray(image) for image in images]

    # im = Image.fromarray(generated_image)
    # im.save("./test_images/output.jpeg")

    # print(generated_image.shape)


if __name__ == "__main__":
    main("hf_inpaint")
