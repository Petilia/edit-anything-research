# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import numpy as np
import cv2
import tritonclient.http as httpclient
from mobile_sam.utils.transforms import ResizeLongestSide


SAVE_INTERMEDIATE_IMAGES = False

def draw_mask(img, mask, color=(0,255,0)):
    # color to fill
    color = np.array(color, dtype='uint8')

    # equal color where mask, else image
    # this would paint your object silhouette entirely with `color`
    masked_img = np.where(mask[..., None], color, img)

    # use `addWeighted` to blend the two images
    # the object will be tinted toward `color`
    return cv2.addWeighted(img, 0.8, masked_img, 0.2,0)

def resize_with_pad(image: np.array,
                    new_shape: tuple[int, int],
                    padding_color: tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image


def resize_longest(image: np.array,
                   new_shape: int) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(new_shape)/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    return image


# def mask_postprocessing(self, masks: torch.Tensor, orig_im_size: torch.Tensor) -> torch.Tensor:
#     masks = F.interpolate(
#         masks,
#         size=(self.img_size, self.img_size),
#         mode="bilinear",
#         align_corners=False,
#     )

#     prepadded_size = self.resize_longest_image_size(orig_im_size, self.img_size).to(torch.int64)
#     masks = masks[..., : prepadded_size[0], : prepadded_size[1]]  # type: ignore

#     orig_im_size = orig_im_size.to(torch.int64)
#     h, w = orig_im_size[0], orig_im_size[1]
#     masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
#     return masks


def pad_upper_left(img, x=1024):
    base = np.zeros((x, x, 3))
    h, w, c = img.shape
    base[:h, :w, :] = img
    return base



if __name__ == "__main__":
    # Setting up client
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Read image and create input object
    raw_image = cv2.imread("./picture2.jpg")
    orig_im_size = np.array(raw_image.shape[:2])
    # preprocessed_image = raw_image
    # preprocessed_image = resize_with_pad(raw_image, new_shape=(1024,1024))
    transform = ResizeLongestSide(target_length=1024)
    preprocessed_image = transform.apply_image(raw_image)

    raw_point = np.array([450, 375])
    # preprocessed_point = raw_point
    preprocessed_point = transform.apply_coords(raw_point, orig_im_size)
    image_with_point = cv2.circle(preprocessed_image, preprocessed_point.astype(int), radius=5, color=(0, 255, 0), thickness=2)
    cv2.imwrite("image_with_point.png", image_with_point)

    preprocessed_image = preprocessed_image.astype(np.float32)

    print("image shape", preprocessed_image.shape)

    detection_input = httpclient.InferInput(
        "input_image", preprocessed_image.shape, datatype="FP32"
    )
    detection_input.set_data_from_numpy(preprocessed_image, binary_data=True)

    # Query the server
    decoder_response = client.infer(
        model_name="sam_encoder", inputs=[detection_input]
    )

    # Process responses from detection model
    embeddings = decoder_response.as_numpy("image_embeddings")
    print("embeddings shape", embeddings.shape)
    np.save("embeddings.npy", embeddings)


    embeddings_input = httpclient.InferInput(
        "image_embeddings", embeddings.shape, datatype="FP32"
    )
    embeddings_input.set_data_from_numpy(embeddings, binary_data=True)

    point_coords = np.array([[preprocessed_point, [0.0, 0.0]]], dtype=np.float32)
    point_coords_input = httpclient.InferInput(
        "point_coords", point_coords.shape, datatype="FP32"
    )
    point_coords_input.set_data_from_numpy(point_coords, binary_data=True)

    point_labels = np.array([[1, -1]], dtype=np.float32)
    point_labels_input = httpclient.InferInput(
        "point_labels", point_labels.shape, datatype="FP32"
    )
    point_labels_input.set_data_from_numpy(point_labels, binary_data=True)

    # mask_input = np.array([0], dtype=np.float32)
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    mask_input_input = httpclient.InferInput(
        "mask_input", mask_input.shape, datatype="FP32"
    )
    mask_input_input.set_data_from_numpy(mask_input, binary_data=True)

    has_mask_input = np.array([0], dtype=np.float32)
    has_mask_input_input = httpclient.InferInput(
        "has_mask_input", has_mask_input.shape, datatype="FP32"
    )
    has_mask_input_input.set_data_from_numpy(has_mask_input, binary_data=True)

    # orig_im_size = np.array(preprocessed_image.shape[:2], dtype=np.float32)
    orig_im_size = orig_im_size.astype(np.float32)
    orig_im_size_input = httpclient.InferInput(
        "orig_im_size", orig_im_size.shape, datatype="FP32"
    )
    orig_im_size_input.set_data_from_numpy(orig_im_size, binary_data=True)


    # Query the server
    decoder_response = client.infer(
        model_name="sam_decoder", inputs=[embeddings_input, point_coords_input, point_labels_input, mask_input_input, has_mask_input_input, orig_im_size_input]
    )

    # Process responses from detection model
    masks = decoder_response.as_numpy("low_res_masks")
    print("masks shape", masks.shape)

    mask = cv2.resize(masks[0,0,:,:], dsize=preprocessed_image.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    mask = (mask > 0)*255

    cv2.imwrite("mask.png", mask)
    cv2.imwrite("image_with_mask.png", draw_mask(preprocessed_image, mask, color=(0, 255, 0)))
