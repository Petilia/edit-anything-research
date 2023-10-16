from flask import Flask, render_template, request, jsonify, send_file,redirect
from flask_cors import CORS
from typing import Any, Dict, List
from utils import mkdir_or_exist
from collections import deque
import threading
import queue
import cv2
import numpy as np
import io, os
import base64
import requests
import pickle
import json
from PIL import Image
import io


class Mode:
    def __init__(self) -> None:
        self.IMAGE = 1
        self.MASKS = 2
        self.CLEAR = 3
        self.P_POINT = 4
        self.UNDO = 8
        self.COLOR_MASKS = 9
        self.INPAINT = 10
        self.DELETE_OBJ = 11
        

MODE = Mode()

class SAM_Web_App:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
       
                    
        self.optimize = True
        self.prompt = ''

        print("Done")

        # Store the image globally on the server
        self.origin_image = None
        self.processed_img = None
        self.masked_img = None
        self.image_with_square = None
        self.colorMasks = None
        self.imgSize = None
        self.user_id = None           # To run self.predictor.set_image() or not

        self.mode = "p_point"           # p_point / n_point / box
        self.curr_view = "image"
        self.queue = deque(maxlen=1000)  # For undo list
        self.prev_inputs = deque(maxlen=500)

        self.points = []
        self.points_label = []
        self.masks = []

        # Set the default save path to the Downloads folder
        home_dir = os.path.expanduser("~")
        self.save_path = os.path.join(home_dir, "Downloads")

        self.app.route('/', methods=['GET'])(self.home)
        self.app.route('/set_prompt', methods=['POST'])(self.set_prompt)
        self.app.route('/upload_image', methods=['POST'])(self.upload_image)
        self.app.route('/button_click', methods=['POST'])(self.button_click)
        self.app.route('/point_click', methods=['POST'])(self.handle_mouse_click)
  
        self.app.route('/set_save_path', methods=['POST'])(self.set_save_path)
        self.app.route('/save_image', methods=['POST'])(self.save_image)
        self.app.route('/send_stroke_data', methods=['POST'])(self.handle_stroke_data)
        self.app.route('/upload_image_to_another_server', methods=['POST'])(self.upload_image_to_another_server)
        self.app.route('/inpaint', methods=['POST'])(self.inpaint)
        self.app.route('/delete_obj', methods=['POST'])(self.delete_obj)
        
    def home(self): 
         return render_template('index.html', default_save_path=self.save_path)
    
    def set_prompt(self):
        prompt = request.form.get('prompt') 
        if prompt is not None:
            self.prompt = prompt
            print(f"Set prompt: {self.prompt}")
            return jsonify({"status": "success", "message": "prompt set successfully", 'prompt':prompt})
      
        return jsonify({"status": "error", "message": "Invalid prompt"}), 400
            
    
    def set_save_path(self):
        self.save_path = request.form.get("save_path")
        # Perform your server-side checks on the save_path here
        # e.g., check if the path exists, if it is writable, etc.
        if os.path.isdir(self.save_path):
            print(f"Set save path to: {self.save_path}")
            return jsonify({"status": "success", "message": "Save path set successfully"})
        else:
            return jsonify({"status": "error", "message": "Invalid save path"}), 400
        
    def save_image(self):
        # Save the colorMasks
        filename = request.form.get("filename")
        if filename == "":
            return jsonify({"status": "error", "message": "No image to save"}), 400
        print(f"Saving: {filename} ...", end="")
        dirname = os.path.join(self.save_path, filename)
        mkdir_or_exist(dirname)
        # Get the number of existing files in the save_folder
        num_files = len([f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))])
        # Create a unique file name based on the number of existing files
        savename = f"{num_files}.jpg"
        save_path = os.path.join(dirname, savename)
        try:
            encoded_img = cv2.imencode(".jpg", self.colorMasks)[1]
            encoded_img.tofile(save_path)
            print("Done!")
            return jsonify({"status": "success", "message": f"Image saved to {save_path}"})
        except:
            return jsonify({"status": "error", "message": "Imencode error"}), 400


    def upload_image_to_another_server(self):
        try:
            url = "http://185.213.209.37:8012/upload_img/"
            encoded_img = cv2.imencode(".jpg", self.origin_image)[1].tobytes()
            print("Done!") 
            files = {'file': encoded_img}
            response = requests.post(url, files=files)
            self.user_id = response.json()['user_id']

            if response.status_code == 200:
                print("Image uploaded successfully!")
            else:
                print("Error uploading image:", response.text)
            return jsonify({"status": "success", "message": "Image uploaded successfully!"})
        except:
            return jsonify({"status": "error", "message": "Imencode error"}), 400
            
            
    def upload_image(self):
        if 'image' not in request.files:
            return jsonify({'error': 'No image in the request'}), 400

        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Store the image globally
        self.origin_image = image
        self.processed_img = image
        self.masked_img = np.zeros_like(image)
        self.colorMasks = np.zeros_like(image)
        self.imgSize = image.shape

        self.upload_image_to_another_server()

        # Reset inputs and masks and image ebedding
        self.reset_inputs()
        self.reset_masks()
        self.queue.clear()
        self.prev_inputs.clear()

        return "Uploaded image, successfully initialized"

    def button_click(self):
        if self.processed_img is None:
            return jsonify({'error': 'No image available for processing'}), 400

        data = request.get_json()
        button_id = data['button_id']
        print(f"Button {button_id} clicked")

        # Info
        info = {
            'event': 'button_click',
            'data': button_id
        }

        # Process and return the image
        return self.process_image(self.processed_img, info)

    def handle_mouse_click(self):
        if self.processed_img is None:
            return jsonify({'error': 'No image available for processing'}), 400

        data = request.get_json()
        x = data['x']
        y = data['y']
        print(f'Point clicked at: {x}, {y}')
        self.points.append(np.array([x, y], dtype=np.float32))
        self.points_label.append(1 if self.mode == 'p_point' else 0)

        # Add command to queue list
        self.queue.append("point")

        points = np.array(self.points)

        # Отправляем точки на сервер 
        print("id user ",self.user_id)
        response = requests.post(url = f'http://185.213.209.37:8012/upload_points/{self.user_id}', json={
            'point': np.vstack(points).tolist(),
        })
        print('points', np.vstack(points).tolist())
    
        mask = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
        print('type', type(mask))
        self.masks.append({
            "mask": mask,
            "opt": "positive"
            })
        
        self.post_processing(self.masks)
        
        # Update masks image to show
        overlayImage, maskedImage = self.updateMaskImg(self.origin_image, self.masks)
        self.processed_img = overlayImage
        self.masked_img = maskedImage
        return f"Click at image pos {x}, {y}"


    def handle_stroke_data(self):
        data = request.get_json()
        stroke_data = data['stroke_data']

        print("Received stroke data")

        if len(stroke_data) == 0:
            pass
        else:
            # Process the stroke data here
            stroke_img = np.zeros_like(self.origin_image)
            print(f"stroke data len: {len(stroke_data)}")

            latestData = stroke_data[len(stroke_data) - 1]
            strokes, size = latestData['Stroke'], latestData['Size']
            BGRcolor = (latestData['Color']['b'], latestData['Color']['g'], latestData['Color']['r'])
            Rpos, Bpos = 2, 0
            stroke_data_cv2 = []
            for stroke in strokes:
                stroke_data_cv2.append((int(stroke['x']), int(stroke['y'])))
            for i in range(len(strokes) - 1):
                cv2.line(stroke_img, stroke_data_cv2[i], stroke_data_cv2[i + 1], BGRcolor, size)

            if BGRcolor[0] == 255:
                mask = np.squeeze(stroke_img[:, :, Bpos] == 0)
                opt = "negative"
            else: # np.where(BGRcolor == 255)[0] == Rpos
                mask = np.squeeze(stroke_img[:, :, Rpos] > 0)
                opt = "positive"

            self.masks.append({
                "mask": mask,
                "opt": opt
            })

        self.get_colored_masks_image()
        self.processed_img, maskedImage = self.updateMaskImg(self.origin_image, self.masks)
        self.masked_img = maskedImage
        self.queue.append("brush")

        if self.curr_view == "masks":
            print("view masks")
            processed_image = self.masked_img
        elif self.curr_view == "colorMasks":
            print("view color")
            processed_image = self.colorMasks
        else:   # self.curr_view == "image":
            print("view image")
            processed_image = self.processed_img

        _, buffer = cv2.imencode('.jpg', processed_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': img_base64})
    
    def inpaint(self):
        response = requests.get(f"http://185.213.209.37:8012/get_image/{self.user_id}", params={'prompt': self.prompt})
    
        # Check if the request was successful
        if response.status_code == 200:
            # Convert image content to NumPy array
            nparr = np.frombuffer(response.content, np.uint8)
        
            # Decode the NumPy array as an image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.origin_image = image
            self.processed_image = image
            print("Image saved as a NumPy array!")
        else:
            print("Failed to download the image.")

        return image

    def delete_obj(self):
        self.prompt = " "
        response = requests.get(f"http://185.213.209.37:8012/get_image/{self.user_id}", params={'prompt': self.prompt})
    
        # Check if the request was successful
        if response.status_code == 200:
            # Convert image content to NumPy array
            nparr = np.frombuffer(response.content, np.uint8)
        
            # Decode the NumPy array as an image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.origin_image = image
            self.processed_image = image
        else:
            print("Failed to download the image.")

        return image

    def process_image(self, image, info):
        processed_image = image

        if info['event'] == 'button_click':
            id = info['data']
            if (id == MODE.IMAGE):
                self.curr_view = "image"
                processed_image = self.processed_img
            elif (id == MODE.MASKS):
                self.curr_view = "masks"
                processed_image = self.masked_img
            elif (id == MODE.COLOR_MASKS):
                self.curr_view = "colorMasks"
                processed_image = self.colorMasks
            elif (id == MODE.CLEAR):
                processed_image = self.origin_image
                self.processed_img = self.origin_image
                self.reset_inputs()
                self.reset_masks()  
                self.queue.clear()
                self.prev_inputs.clear()
            elif (id == MODE.P_POINT):
                self.mode = "p_point"
          
            elif (id == MODE.INPAINT):
                print('inpaint')
                processed_image = self.inpaint()
                self.origin_image = processed_image
                self.processed_img = processed_image
            elif (id == MODE.DELETE_OBJ):
                self.delete_obj()
            elif (id == MODE.UNDO):
                if len(self.queue) != 0:
                    command = self.queue.pop()
                    command = command.split('-')
                else:
                    command = None
                print(f"Undo {command}")

                if command is None:
                    pass
                elif command[0] == "point":
                    self.points.pop()
                
                if self.curr_view == "masks":
                    print("view masks")
                    processed_image = self.masked_img
                elif self.curr_view == "colorMasks":
                    print("view color")
                    processed_image = self.colorMasks
                else:   # self.curr_view == "image":
                    print("view image")
                    processed_image = self.processed_img
            

        _, buffer = cv2.imencode('.jpg', processed_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': img_base64})
    
    
    def post_processing(self, masks):
        for mask in masks:
            mask_u8 = mask['mask'].astype(np.uint8)
            contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) 
            label = np.uint8(np.zeros((mask_u8.shape[0], mask_u8.shape[1])))
            if len(contours)!=1:
                for num, cnt in enumerate(contours):
                    contour_area = cv2.contourArea(cnt)
                    if contour_area > 130: 
                        if hierarchy[0, num][3] != -1:
                            label = cv2.drawContours(label, [cnt], -1, (0,0,0), -1)

                        else:
                            label = cv2.drawContours(label, [cnt], -1, (255,255,255), -1)
            else:
                label = cv2.drawContours(label, contours, -1, (255,255,255), -1)
                            
            mask['mask'] = label.astype(np.bool_)
    

    def updateMaskImg(self, image, masks):

        if (len(masks) == 0 or masks[0] is None):
            print(masks)
            return image, np.zeros_like(image)
        
        union_mask = np.zeros_like(image)[:, :, 0]
        print(union_mask.shape)
        np.random.seed(0)
        
        for i in range(len(masks)):
            if masks[i]['opt'] == "negative":
                image = self.clearMaskWithOriginImg(self.origin_image, image, masks[i]['mask'])
                union_mask = np.bitwise_and(union_mask, masks[i]['mask'])
            else:
                
                colored = True
                image = self.overlay_mask(image, masks[i]['mask'], 0.9, colored)
                union_mask = np.bitwise_or(union_mask, masks[i]['mask'])
        
        # Cut out objects using union mask
        masked_image = self.origin_image * union_mask[:, :, np.newaxis]
        return image, masked_image
    

    # Function to overlay a mask on an image
    def overlay_mask(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        alpha: float, 
        colored: bool = False,
    ) -> np.ndarray:
        """ Draw mask on origin image

        parameters:
        image:  Origin image
        mask:   Mask that have same size as image
        color:  Mask's color in BGR
        alpha:  Transparent ratio from 0.0-1.0

        return:
        blended: masked image
        """
        # Blend the image and the mask using the alpha value
        if colored:
            color = np.array([0.5, 0.25, 0.25])
        else:
            color = np.array([1.0, 1.0, 1.0])    # BGR
        h, w = mask.shape[-2:]
        mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask *= 255 * alpha
        mask = mask.astype(dtype=np.uint8)
        blended = cv2.add(image, mask)
        
        return blended
    
    def get_colored_masks_image(self):
        masks = self.masks
        darkImg = np.zeros_like(self.origin_image)
        image = darkImg.copy()

        np.random.seed(0)
        if (len(masks) == 0):
            self.colorMasks = image
            return image
        for mask in masks:
            if mask['opt'] == "negative":
                image = self.clearMaskWithOriginImg(darkImg, image, mask['mask'])
            else:
                colored = False
                image = self.overlay_mask(image, mask['mask'], 1, colored)

        self.colorMasks = image
        return image

         
    def clearMaskWithOriginImg(self, originImage, image, mask):
        originImgPart = originImage * np.invert(mask)[:, :, np.newaxis]
        image = image * mask[:, :, np.newaxis]
        image = cv2.add(image, originImgPart)
        return image
    
    def reset_inputs(self):
        self.points = []
        self.points_label = []


    def reset_masks(self):
        self.masks = []
        self.masked_img = np.zeros_like(self.origin_image)
        self.colorMasks = np.zeros_like(self.origin_image)
        
        
    def run(self, debug=True):
        self.app.run(debug=debug, port=80)


if __name__ == '__main__':
    
    app = SAM_Web_App()
   
    
    app.run(debug=True)