import cv2
import torch
import scipy.special
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum
from scipy.spatial.distance import cdist
import time

from model.model import parsingNet

lane_colors = [(0,0,255),(0,255,0),(255,0,0)]

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
			272, 276, 280, 284]



class ModelConfig():

	def __init__(self):
		self.imgWidth = 1280
		self.imgHeight = 720
		self.row_anchor = tusimple_row_anchor
		self.griding_num = 100
		self.cls_num_per_lane = 56

class LaneDetection():
    def __init__(self, model_path, useGPU=False):

        self.useGPU = useGPU

        # Load model configuration based on the model type
        self.cfg = ModelConfig()

        # Initialize model
        self.model = self.buildModel(model_path, self.cfg, useGPU)

        # Initialize image transformation
        self.img_transform = self.imageTransformation()
    
    def buildModel(self,model_path, cfg, useGPU):
        # Load the model architecture
        net = parsingNet(pretrained = False, backbone='18', cls_dim = (cfg.griding_num+1,cfg.cls_num_per_lane,4))


        # Load the weights from the downloaded model
        if useGPU:
            net = net.cuda()
            state_dict = torch.load(model_path, map_location='cuda')['model'] # CUDA
        else:
            state_dict = torch.load(model_path, map_location='cpu')['model'] # CPU

        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        # Load the weights into the model
        net.load_state_dict(compatible_state_dict, strict=False)
        net.eval()

        return net

    def imageTransformation(self):
		# Create transfom operation to resize and normalize the input images
        img_transforms = transforms.Compose([
			transforms.Resize((288, 800)),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])

        return img_transforms

    def detectLanes(self, image, draw_points=True):

        input_tensor = self.preprocess(image)

        # Perform Ai on img
        with torch.no_grad():
            output = self.model(input_tensor)

        # Process output data
        self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)


        # Draw depth image
        visualization_img = self.drawLanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points)

        return visualization_img

    def preprocess(self, img):
        # Transform the image for inference
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        input_img = self.img_transform(img_pil)
        input_tensor = input_img[None, ...]

        if self.useGPU:
            if not torch.backends.mps.is_built():
                input_tensor = input_tensor.cuda()

        return input_tensor

    def process_output(self,output, cfg):		
        # Parse the output of the model
        processed_output = output[0].data.cpu().numpy()
        processed_output = processed_output[:, ::-1, :]
        prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        processed_output = np.argmax(processed_output, axis=0)
        loc[processed_output == cfg.griding_num] = 0
        processed_output = loc


        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        lanes_points = []
        lanes_detected = []

        max_lanes = processed_output.shape[1]
        for lane_num in range(max_lanes):
            lane_points = []
            # Check if there are any points detected in the lane
            if np.sum(processed_output[:, lane_num] != 0) > 2:

                lanes_detected.append(True)

                # Process the first 26 points of each lane
                for point_num in range(processed_output.shape[0]):
                    if point_num > 26:
                        pass
                    else:
                        if processed_output[point_num, lane_num] > 0:
                            lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.imgWidth / 800) - 1, int(cfg.imgHeight * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/288)) - 1 ]
                            lane_points.append(lane_point)
                            
            else:
                lanes_detected.append(False)

            lanes_points.append(lane_points)
        return np.array(lanes_points), np.array(lanes_detected)
        
    def drawLanes(self,input_img, lanes_points, lanes_detected, cfg, draw_points=True):
        # Write the detected line points in the image
        visualization_img = cv2.resize(input_img, (cfg.imgWidth, cfg.imgHeight), interpolation = cv2.INTER_AREA)

        # Draw a mask for the current lane
        if(lanes_detected[1] and lanes_detected[2]):
            lane_segment_img = visualization_img.copy()
            
            cv2.fillPoly(lane_segment_img, pts = [np.vstack((lanes_points[1],np.flipud(lanes_points[2])))], color =(255,191,0))
            visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)

        if(draw_points):
            for lane_num,lane_points in enumerate(lanes_points):
                if lane_num > 2:
                    break
                for lane_point in lane_points:
                    cv2.circle(visualization_img, (lane_point[0],lane_point[1]), 3, lane_colors[lane_num], -1)

        return visualization_img