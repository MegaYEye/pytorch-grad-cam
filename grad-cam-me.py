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
preprocessed_img = preprocessed_img.cuda()
model=models.vgg19(pretrained=True).cuda()
# set on evaluate model. i.e. Dropout layer.
model.eval()
# module is a sequential model.
def get_grad(x, filter_layers=["35"]):
    res=[]
    for name, module in model.features._modules.items():
        x = module(x)
        if name in filter_layers:
            x.register_hook(lambda grad: res.append(grad))
    return res, x
# print(model(preprocessed_img).shape)
grad,x = get_grad(preprocessed_img)
print(x.shape)
output = model.classifier(x.view(x.size(0), -1))
index = np.argmax(output.cpu().data.numpy())
one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
one_hot[0][index] = 1
one_hot = torch.Tensor(one_hot)
print(one_hot.shape)
# torch.Size([1, 1000])

         