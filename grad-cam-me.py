import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import os
import logging
logging.basicConfig(level=logging.DEBUG)
image_path =os.path.join( os.getcwd(), "a/both.png"  )
img = cv2.imread(image_path, 1)
img = np.float32(cv2.resize(img, (224, 224))) / 255
logging.debug( "now img size is: " + str(img.shape)  )
logging.debug("now start preprocessing")
preprocessed_img = img.copy()[: , :, ::-1]
means=[0.485, 0.456, 0.406]
stds=[0.229, 0.224, 0.225]

for i in range(3):
	preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
	preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
# turn to float..
preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
preprocessed_img = torch.from_numpy(preprocessed_img)
# torch.Size([3, 224, 224]) -> torch.Size([1, 3, 224, 224])
preprocessed_img.unsqueeze_(0)
preprocessed_img.requires_grad=True

model=models.vgg19(pretrained=True)
# set on evaluate model. i.e. Dropout layer.
model.eval()
# module is a sequential model.
def get_grad(x, filter_layers=[]):
    res=[]
    for name, module in model._modules.items():
        x = module(x)
        if name in filter_layers:
            x.register_hook(lambda grad: res.append(grad))
    return res
         