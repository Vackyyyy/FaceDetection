import os
from functools import reduce

import torch
import torch.nn as nn

# from .mobilenetv2 import MobileNetV2


class BaseBackbone(nn.Module):
    """ Superclass of Replaceable Backbone Model for Semantic Estimation
    """

    def __init__(self, in_channels):
        super(BaseBackbone, self).__init__()
        self.in_channels = in_channels

        self.model = None
        self.enc_channels = []

    def forward(self, x):
        raise NotImplementedError

    def load_pretrained_ckpt(self):
        raise NotImplementedError


class MobileNetV2Backbone(BaseBackbone):
    """ MobileNetV2 Backbone 
    """

    def __init__(self, in_channels):
        super(MobileNetV2Backbone, self).__init__(in_channels)

        self.model = MobileNetV2(self.in_channels, alpha=1.0, expansion=6, num_classes=None)
        self.enc_channels = [16, 24, 32, 96, 1280]

    def forward(self, x):
        # x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        enc2x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(2, 4)), x)
        x = self.model.features[2](x)
        x = self.model.features[3](x)
        enc4x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(4, 7)), x)
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = self.model.features[6](x)
        enc8x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(7, 14)), x)
        x = self.model.features[7](x)
        x = self.model.features[8](x)
        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        x = self.model.features[13](x)
        enc16x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(14, 19)), x)
        x = self.model.features[14](x)
        x = self.model.features[15](x)
        x = self.model.features[16](x)
        x = self.model.features[17](x)
        x = self.model.features[18](x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch 
        ckpt_path = './pretrained/mobilenetv2_human_seg.ckpt'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained mobilenetv2 backbone')
            exit()
        
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt)


SUPPORTED_BACKBONES = {
    'mobilenetv2': MobileNetV2Backbone,
}

""" This file is adapted from https://github.com/thuyngch/Human-Segmentation-PyTorch"""

import math
import json
from functools import reduce

import torch
from torch import nn


#------------------------------------------------------------------------------
#  Useful functions
#------------------------------------------------------------------------------

def _make_divisible(v, divisor, min_value=None):
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


def conv_bn(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU6(inplace=True)
	)


def conv_1x1_bn(inp, oup):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU6(inplace=True)
	)


#------------------------------------------------------------------------------
#  Class of Inverted Residual block
#------------------------------------------------------------------------------

class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, expansion, dilation=1):
		super(InvertedResidual, self).__init__()
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = round(inp * expansion)
		self.use_res_connect = self.stride == 1 and inp == oup

		if expansion == 1:
			self.conv = nn.Sequential(
				# dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)
		else:
			self.conv = nn.Sequential(
				# pw
				nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)

	def forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)


#------------------------------------------------------------------------------
#  Class of MobileNetV2
#------------------------------------------------------------------------------

class MobileNetV2(nn.Module):
	def __init__(self, in_channels, alpha=1.0, expansion=6, num_classes=1000):
		super(MobileNetV2, self).__init__()
		self.in_channels = in_channels
		self.num_classes = num_classes
		input_channel = 32
		last_channel = 1280
		interverted_residual_setting = [
			# t, c, n, s
			[1        , 16, 1, 1],
			[expansion, 24, 2, 2],
			[expansion, 32, 3, 2],
			[expansion, 64, 4, 2],
			[expansion, 96, 3, 1],
			[expansion, 160, 3, 2],
			[expansion, 320, 1, 1],
		]

		# building first layer
		input_channel = _make_divisible(input_channel*alpha, 8)
		self.last_channel = _make_divisible(last_channel*alpha, 8) if alpha > 1.0 else last_channel
		self.features = [conv_bn(self.in_channels, input_channel, 2)]

		# building inverted residual blocks
		for t, c, n, s in interverted_residual_setting:
			output_channel = _make_divisible(int(c*alpha), 8)
			for i in range(n):
				if i == 0:
					self.features.append(InvertedResidual(input_channel, output_channel, s, expansion=t))
				else:
					self.features.append(InvertedResidual(input_channel, output_channel, 1, expansion=t))
				input_channel = output_channel

		# building last several layers
		self.features.append(conv_1x1_bn(input_channel, self.last_channel))

		# make it nn.Sequential
		self.features = nn.Sequential(*self.features)

		# building classifier
		if self.num_classes is not None:
			self.classifier = nn.Sequential(
				nn.Dropout(0.2),
				nn.Linear(self.last_channel, num_classes),
			)

		# Initialize weights
		self._init_weights()

	def forward(self, x):
		# Stage1
		x = self.features[0](x)
		x = self.features[1](x)
		# Stage2
		x = self.features[2](x)
		x = self.features[3](x)
		# Stage3
		x = self.features[4](x)
		x = self.features[5](x)
		x = self.features[6](x)
		# Stage4
		x = self.features[7](x)
		x = self.features[8](x)
		x = self.features[9](x)
		x = self.features[10](x)
		x = self.features[11](x)
		x = self.features[12](x)
		x = self.features[13](x)
		# Stage5
		x = self.features[14](x)
		x = self.features[15](x)
		x = self.features[16](x)
		x = self.features[17](x)
		x = self.features[18](x)

		# Classification
		if self.num_classes is not None:
			x = x.mean(dim=(2,3))
			x = self.classifier(x)
			
		# Output
		return x

	def _load_pretrained_model(self, pretrained_file):
		pretrain_dict = torch.load(pretrained_file, map_location='cpu')
		model_dict = {}
		state_dict = self.state_dict()
		print("[MobileNetV2] Loading pretrained model...")
		for k, v in pretrain_dict.items():
			if k in state_dict:
				model_dict[k] = v
			else:
				print(k, "is ignored")
		state_dict.update(model_dict)
		self.load_state_dict(state_dict)

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()


import torch
import torch.nn as nn
import torch.nn.functional as F

# from .backbones import SUPPORTED_BACKBONES
# where is .backbones? It is in the same directory as this file, so it is a relative import.
# This is a relative import because it starts with a dot.



#------------------------------------------------------------------------------
#  MODNet Basic Modules
#------------------------------------------------------------------------------

class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)
        
    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride=stride, padding=padding, dilation=dilation, 
                      groups=groups, bias=bias)
        ]

        if with_ibn:       
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) 


class SEBlock(nn.Module):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf 
    """

    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)


#------------------------------------------------------------------------------
#  MODNet Branches
#------------------------------------------------------------------------------

class LRBranch(nn.Module):
    """ Low Resolution Branch of MODNet
    """

    def __init__(self, backbone):
        super(LRBranch, self).__init__()

        enc_channels = backbone.enc_channels
        
        self.backbone = backbone
        self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
        self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)
        self.conv_lr8x = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)
        self.conv_lr = Conv2dIBNormRelu(enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False)

    def forward(self, img, inference):
        enc_features = self.backbone.forward(img)
        enc2x, enc4x, enc32x = enc_features[0], enc_features[1], enc_features[4]

        enc32x = self.se_block(enc32x)
        lr16x = F.interpolate(enc32x, scale_factor=2, mode='bilinear', align_corners=False)
        lr16x = self.conv_lr16x(lr16x)
        lr8x = F.interpolate(lr16x, scale_factor=2, mode='bilinear', align_corners=False)
        lr8x = self.conv_lr8x(lr8x)

        pred_semantic = None
        if not inference:
            lr = self.conv_lr(lr8x)
            pred_semantic = torch.sigmoid(lr)

        return pred_semantic, lr8x, [enc2x, enc4x] 


class HRBranch(nn.Module):
    """ High Resolution Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels):
        super(HRBranch, self).__init__()

        self.tohr_enc2x = Conv2dIBNormRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_enc2x = Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=2, padding=1)

        self.tohr_enc4x = Conv2dIBNormRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)

        self.conv_hr4x = nn.Sequential(
            Conv2dIBNormRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr2x = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, enc2x, enc4x, lr8x, inference):
        img2x = F.interpolate(img, scale_factor=1/2, mode='bilinear', align_corners=False)
        img4x = F.interpolate(img, scale_factor=1/4, mode='bilinear', align_corners=False)

        enc2x = self.tohr_enc2x(enc2x)
        hr4x = self.conv_enc2x(torch.cat((img2x, enc2x), dim=1))

        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.conv_enc4x(torch.cat((hr4x, enc4x), dim=1))

        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        hr4x = self.conv_hr4x(torch.cat((hr4x, lr4x, img4x), dim=1))

        hr2x = F.interpolate(hr4x, scale_factor=2, mode='bilinear', align_corners=False)
        hr2x = self.conv_hr2x(torch.cat((hr2x, enc2x), dim=1))

        pred_detail = None
        if not inference:
            hr = F.interpolate(hr2x, scale_factor=2, mode='bilinear', align_corners=False)
            hr = self.conv_hr(torch.cat((hr, img), dim=1))
            pred_detail = torch.sigmoid(hr)

        return pred_detail, hr2x


class FusionBranch(nn.Module):
    """ Fusion Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels):
        super(FusionBranch, self).__init__()
        self.conv_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)
        
        self.conv_f2x = Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, lr8x, hr2x):
        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        lr2x = F.interpolate(lr4x, scale_factor=2, mode='bilinear', align_corners=False)

        f2x = self.conv_f2x(torch.cat((lr2x, hr2x), dim=1))
        f = F.interpolate(f2x, scale_factor=2, mode='bilinear', align_corners=False)
        f = self.conv_f(torch.cat((f, img), dim=1))
        pred_matte = torch.sigmoid(f)

        return pred_matte


#------------------------------------------------------------------------------
#  MODNet
#------------------------------------------------------------------------------

class MODNet(nn.Module):
    """ Architecture of MODNet
    """

    def __init__(self, in_channels=3, hr_channels=32, backbone_arch='mobilenetv2', backbone_pretrained=True):
        super(MODNet, self).__init__()

        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained

        self.backbone = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels)

        self.lr_branch = LRBranch(self.backbone)
        self.hr_branch = HRBranch(self.hr_channels, self.backbone.enc_channels)
        self.f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        if self.backbone_pretrained:
            self.backbone.load_pretrained_ckpt()                

    def forward(self, img, inference):
        pred_semantic, lr8x, [enc2x, enc4x] = self.lr_branch(img, inference)
        pred_detail, hr2x = self.hr_branch(img, enc2x, enc4x, lr8x, inference)
        pred_matte = self.f_branch(img, lr8x, hr2x)

        return pred_semantic, pred_detail, pred_matte
    
    def freeze_norm(self):
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    continue

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
import torch
import cv2
from torchvision import transforms
from facenet_pytorch import MTCNN
from pathlib import Path
# MODNet imports
# from app.MODNet.src.models.modnet import MODNet
import os

app = FastAPI()

# Mount static files (e.g., for serving uploaded images)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Load MODNet model on CPU
def load_modnet_model(ckpt_path):
    modnet = MODNet(backbone_pretrained=False)
    modnet = torch.nn.DataParallel(modnet)
    modnet.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    modnet.eval()
    return modnet

# Preprocessing function
def preprocess_image(image):
    original_size = image.size
    image = image.resize((512, 512), Image.BILINEAR)
    image = transforms.ToTensor()(image).unsqueeze(0)
    return image, original_size

# Postprocessing function
def postprocess_alpha(alpha, original_size):
    alpha = alpha.squeeze().cpu().numpy()
    alpha = cv2.resize(alpha, original_size, interpolation=cv2.INTER_LINEAR)
    return alpha

# Function to refine the alpha matte and smooth the edges
def refine_alpha(alpha):
    # Apply a binary threshold to the alpha to remove weak regions
    alpha_threshold = (alpha > 0.1).astype(np.uint8) * 255

    # Use GaussianBlur to smooth the edges of the alpha mask
    alpha_blurred = cv2.GaussianBlur(alpha_threshold, (7, 7), 0)

    # Normalize the alpha values back between 0 and 1 after blurring
    alpha_smoothed = alpha_blurred / 255.0

    return alpha_smoothed

# Function to crop the above neck region with face detection
def crop_above_neck(image, alpha):
    # Convert image from PIL to NumPy without changing the color space (keep it RGB)
    original_image = np.array(image)  # Image is still in RGB
    alpha = refine_alpha(alpha)
    alpha = (alpha * 255).astype(np.uint8)

    # Create a white background (RGB format)
    white_background = np.ones_like(original_image) * 255
    result = np.where(alpha[..., None] > 0, original_image, white_background)

    # Use MTCNN for face detection
    mtcnn = MTCNN(keep_all=False)
    boxes, _ = mtcnn.detect(image)

    # Ensure exactly one face is detected
    if boxes is None or len(boxes) != 1:
        raise HTTPException(status_code=400, detail="The image must contain exactly one face.")

    x1, y1, x2, y2 = boxes[0]
    padding = int((y2 - y1) * 0.3)
    padding_below = int((y2 - y1) * 0.1)
    top_region = max(int(y1 - padding), 0)
    bottom_region = min(int(y2 - padding_below), original_image.shape[0])

    face_and_hair_region = result[top_region:bottom_region, :, :]

    full_size_face_image = np.ones_like(original_image) * 255
    full_size_face_image[top_region:bottom_region, :, :] = face_and_hair_region

    # Resize the output image to 600x600 while keeping RGB format
    resized_image = cv2.resize(full_size_face_image, (600, 600), interpolation=cv2.INTER_LINEAR)

    return resized_image

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    image = Image.open(file.file).convert("RGB")  # Ensure image is in RGB format
    modnet = load_modnet_model('modnet_photographic_portrait_matting.ckpt')

    im, original_size = preprocess_image(image)

    with torch.no_grad():
        _, _, matte = modnet(im, True)

    alpha = postprocess_alpha(matte, original_size)
    processed_image = crop_above_neck(image, alpha)

    # Save the processed image in RGB format
    output_path = Path("app/static/uploads") / f"{file.filename}_processed.jpg"
    cv2.imwrite(str(output_path), cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))  # Keep colors intact

    return {"image_url": f"/static/uploads/{output_path.name}"}

@app.get("/download/{filename}", response_class=FileResponse)
async def download_image(filename: str):
    file_path = Path("app/static/uploads") / filename
    if file_path.exists():
        return file_path
    else:
        raise HTTPException(status_code=404, detail="File not found.")
