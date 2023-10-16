from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from starlette.requests import Request
from transforms import ResizeLongestSide
import tritonclient.http as httpclient
from pydantic import BaseModel
from PIL import Image, ImageOps
from copy import copy
import numpy as np
import cv2
import asyncio

import functools
import contextvars
import aiofiles
import io
import os
import json
from uuid import uuid4

import argparse
import base64
import time
from io import BytesIO

import numpy as np
import requests
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *

# import

# from mobile_sam.utils.transforms import ResizeLongestSide

app = FastAPI()
client = httpclient.InferenceServerClient(url="127.0.0.1:8000")
DEBUG = False


@app.post("/upload_img/")
async def upload_image(file: UploadFile = File(...)):
    try:
        user_id = uuid4()
        print(user_id)
        contents = await file.read()
        async with aiofiles.open(f'./static/images/{user_id}.jpeg', 'wb') as buffer:
            await buffer.write(contents)

        return JSONResponse(content={'user_id': str(user_id)}, status_code=200)

    except Exception as e:
        return HTTPException(detail='OTSOSI: ' + str(e), status_code=500)


async def to_thread(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)


async def save_image_async(path, image):
    await to_thread(cv2.imwrite, path, image)


def draw_mask(img, mask, color=(0, 255, 0)):
    color = np.array(color, dtype='uint8')
    masked_img = np.where(mask[..., None], color, img)
    return cv2.addWeighted(img, 0.8, masked_img, 0.2, 0)


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


def inpaint(prompt, img_path, mask_path, inpainted_img_path):
    model_name = "hf_inpaint"
    image = Image.open(img_path)
    init_width, init_height = image.size

    mask = Image.open(mask_path)
    #mask = ImageOps.invert(mask)
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

    print(input_tensors)
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
    decoded_images = []

    encoded_image = generated_image[0]

    image_data = base64.b64decode(encoded_image)
    image_buffer = BytesIO(image_data)

    img = Image.open(image_buffer).resize((init_width, init_height))
    img.save(inpainted_img_path)


def get_inputs(image_file, point):
    transform = ResizeLongestSide(1024)
    image = cv2.imread(image_file)
    image_bytes = open(image_file, 'rb').read()
    image_transformed = np.array(list(image_bytes), dtype=np.uint8)

    input_point = np.array([point])
    input_label = np.array([1])

    onnx_coord = np.concatenate(
        [input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate(
        [input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

    onnx_coord = transform.apply_coords(
        onnx_coord, image.shape[:2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    return {
        "input_image": image_transformed,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
    }


@ app.post("/upload_points/{user_id}")
async def upload_points(request: Request, user_id: str = None):
    try:
        inp_json = await request.json()

        point = inp_json['point'][-1]
        image_file = f'./static/images/{user_id}.jpeg'

        #image = cv2.imread(image_file)
        inputs = get_inputs(image_file, point)

        # Create encoder input image
        encoder_input = httpclient.InferInput(
            "input_image", inputs["input_image"].shape, datatype="UINT8"
        )
        encoder_input.set_data_from_numpy(
            inputs["input_image"], binary_data=True)
        # Get encoder output embeddings
        encoder_response = client.infer(
            model_name="encoder_ensemble", inputs=[encoder_input]
        )
        image_embeddings = encoder_response.as_numpy("image_embeddings")

        # Create encoder inputs
        # image_embeddings
        embeddings_input = httpclient.InferInput(
            "image_embeddings", image_embeddings.shape, datatype="FP32"
        )
        embeddings_input.set_data_from_numpy(
            image_embeddings, binary_data=True)
        # point_coords
        point_coords_input = httpclient.InferInput(
            "point_coords", inputs["point_coords"].shape, datatype="FP32"
        )
        point_coords_input.set_data_from_numpy(
            inputs["point_coords"], binary_data=True)
        # point_labels
        point_labels_input = httpclient.InferInput(
            "point_labels", inputs["point_labels"].shape, datatype="FP32"
        )
        point_labels_input.set_data_from_numpy(
            inputs["point_labels"], binary_data=True)
        # mask_input
        mask_input_input = httpclient.InferInput(
            "mask_input", inputs["mask_input"].shape, datatype="FP32"
        )
        mask_input_input.set_data_from_numpy(
            inputs["mask_input"], binary_data=True)
        # has_mask_input
        has_mask_input_input = httpclient.InferInput(
            "has_mask_input", inputs["has_mask_input"].shape, datatype="FP32"
        )
        has_mask_input_input.set_data_from_numpy(
            inputs["has_mask_input"], binary_data=True)
        # orig_im_size
        orig_im_size_input = httpclient.InferInput(
            "orig_im_size", inputs["orig_im_size"].shape, datatype="FP32"
        )
        orig_im_size_input.set_data_from_numpy(
            inputs["orig_im_size"], binary_data=True)

        decoder_response = client.infer(
            model_name="sam_decoder", inputs=[embeddings_input, point_coords_input, point_labels_input, mask_input_input, has_mask_input_input, orig_im_size_input]
        )

        masks = decoder_response.as_numpy("masks")
        masks = (masks[0, 0, :, :] > 0) * 255
        # dikate masks
        print(masks.dtype)
        masks = cv2.dilate(masks.astype(np.uint8),
                           np.ones((5, 5), 'uint8'), iterations=5)

        masked_img_path = f"./static/masks/masked_{str(user_id)}.png"
        mask_pth = f"./static/masks/{str(user_id)}.png"

        #save_image_task = asyncio.create_task(save_image_async(masked_img_path, draw_mask(image, masks, color=(0, 255, 0))))
        # await save_image_task

        #save_iamge_task_1 = asyncio.create_task(save_image_async(f"./static/masks/{str(user_id)}.png", masks))
        # await save_iamge_task_1

        cv2.imwrite(mask_pth, masks)
        cv2.imwrite(masked_img_path, draw_mask(
            image, masks, color=(0, 255, 0)))

        points_upload_response = FileResponse(mask_pth, media_type="image/png")
        os.remove(masked_img_path)

        return points_upload_response

        #
    except Exception as e:
        print(e)
        return HTTPException(detail=str(e), status_code=500)


@ app.get("/get_image/{user_id}")
async def get_image(user_id: str=None, prompt: str = ''):
    try:

        img_path = f'./static/images/{user_id}.jpeg'
        inpainted_img_path = f'./static/inpainted_images/{user_id}.jpeg'
        mask_path = f"./static/masks/{str(user_id)}.png"
        inpaint(prompt, img_path, mask_path, inpainted_img_path)

        get_img = FileResponse(inpainted_img_path, media_type="image/jpeg")

        os.remove(img_path)
        os.remove(inpainted_img_path)
        os.remove(mask_path)

        return get_img

    except Exception as e:
        return HTTPException(detail=str(e), status_code=500)


@ app.on_event("shutdown")
async def delete_uploaded_files():
    for folder_path in ['./static/images/', './static/masks/', './static/inpainted_images/']:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8012)
