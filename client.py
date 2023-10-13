import numpy as np
import cv2
import tritonclient.http as httpclient
from transforms import ResizeLongestSide
from copy import copy


def draw_mask(img, mask, color=(0,255,0)):
    # color to fill
    color = np.array(color, dtype='uint8')

    # equal color where mask, else image
    # this would paint your object silhouette entirely with `color`
    masked_img = np.where(mask[..., None], color, img)

    # use `addWeighted` to blend the two images
    # the object will be tinted toward `color`
    return cv2.addWeighted(img, 0.8, masked_img, 0.2,0)

def get_inputs(image_file, point):
    transform = ResizeLongestSide(1024)

    # image_transformed = transform.apply_image(image)
    image = cv2.imread(image_file)
    image_bytes = open(image_file, "rb").read()
    image_transformed = np.array(list(image_bytes), dtype=np.uint8)

    input_point = np.array([point])
    input_label = np.array([1])

    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

    onnx_coord = transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
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

DEBUG = True

if __name__ == "__main__":
    client = httpclient.InferenceServerClient(url="localhost:8000")

    image_name = "./picture2.jpg"
    image = cv2.imread(image_name)
    point = [256, 256]
    inputs = get_inputs(image_name, point)

    if DEBUG:
        image_with_point = cv2.circle(copy(image), np.array(point), radius=5, color=(0, 255, 0), thickness=2)
        cv2.imwrite("image_with_point.png", image_with_point)
    if DEBUG:
        print("original image shape", image.shape)
        print("image shape", inputs["input_image"].shape)


    # Create encoder input image
    encoder_input = httpclient.InferInput(
        "input_image", inputs["input_image"].shape, datatype="UINT8"
    )
    encoder_input.set_data_from_numpy(inputs["input_image"], binary_data=True)
    # Get encoder output embeddings
    encoder_response = client.infer(
        model_name="encoder_ensemble", inputs=[encoder_input]
    )
    image_embeddings = encoder_response.as_numpy("image_embeddings")
    if DEBUG:
        print("embeddings shape", image_embeddings.shape)


    # Create encoder inputs
    ## image_embeddings
    embeddings_input = httpclient.InferInput(
        "image_embeddings", image_embeddings.shape, datatype="FP32"
    )
    embeddings_input.set_data_from_numpy(image_embeddings, binary_data=True)
    ## point_coords
    point_coords_input = httpclient.InferInput(
        "point_coords", inputs["point_coords"].shape, datatype="FP32"
    )
    point_coords_input.set_data_from_numpy(inputs["point_coords"], binary_data=True)
    ## point_labels
    point_labels_input = httpclient.InferInput(
        "point_labels", inputs["point_labels"].shape, datatype="FP32"
    )
    point_labels_input.set_data_from_numpy(inputs["point_labels"], binary_data=True)
    ## mask_input
    mask_input_input = httpclient.InferInput(
        "mask_input", inputs["mask_input"].shape, datatype="FP32"
    )
    mask_input_input.set_data_from_numpy(inputs["mask_input"], binary_data=True)
    ## has_mask_input
    has_mask_input_input = httpclient.InferInput(
        "has_mask_input", inputs["has_mask_input"].shape, datatype="FP32"
    )
    has_mask_input_input.set_data_from_numpy(inputs["has_mask_input"], binary_data=True)
    ## orig_im_size
    orig_im_size_input = httpclient.InferInput(
        "orig_im_size", inputs["orig_im_size"].shape, datatype="FP32"
    )
    orig_im_size_input.set_data_from_numpy(inputs["orig_im_size"], binary_data=True)


    # Get decoder outputs
    decoder_response = client.infer(
        model_name="sam_decoder", inputs=[embeddings_input, point_coords_input, point_labels_input, mask_input_input, has_mask_input_input, orig_im_size_input]
    )

    # Process responses from detection model
    masks = decoder_response.as_numpy("masks")
    if DEBUG:
        print("masks shape", masks.shape)
    masks = (masks[0,0,:,:] > 0)*255

    if DEBUG:
        cv2.imwrite("mask.png", masks)
        cv2.imwrite("image_with_mask.png", draw_mask(image, masks, color=(0, 255, 0)))
