from io import BytesIO
from PIL import Image
import base64
import numpy as np
import torch
import webdataset as wds
import matplotlib as plt
import torchvision.transforms as transforms
import math

from models import cnn, vae #make this work...should be doable
from utils import img_sizes, membership_dict, round_number



'''
So far 3 classes 
1: ScaleImg -> just scales the img and options for other stuff if needed
2: Tars -> creates a list of Tars to use during Testing based off membership_lvl and ws
3: ImgMatch -> trains off Scaled Img and then tests vs Tars return based off membership
'''




class ScaleImg:
    def __init__(self, img):
        self.img = img
        self.fix_image()
        self.get_scale()
        self.scale_img()



    def fix_image(self):
        x = self.img.split(',', 1)
        img_bytes = base64.b64decode(x[1]) 
        image_PIL = Image.open(BytesIO(img_bytes))
        np_img = np.array(image_PIL) #has (x, x, 4).shape
        ii = torch.Tensor(np_img[:, :, 3])
        self.fixed_img = torch.unsqueeze(ii, 0) #fixed (1, x, x)
        self.avg = math.ceil((len(self.fixed_img[0][0]) + len(self.fixed_img[0][:,1])) / 2)



    def get_scale(self):
        if self.avg <= img_sizes['small']['dim']:
            self.size = img_sizes['small']
        elif self.avg <= img_sizes['med']['dim']:
            self.size = img_sizes['med']
        else:
            self.size = img_sizes['large']




    def scale_img(self):
        x = self.size['dim']
        resize_img = transforms.Resize(size=(x, x))
        self.scaled_img = resize_img(self.fixed_img) 





class FileName:
    """
    used when creating Tars
    and again when they submit and creating a dict of files to test on along with slots for topk
    """
    def __init__(self, currency, tf, ws):
        self.currency = currency
        self.tf = tf 
        self.ws = ws 
        self.name = ('{}_{}_{}'.format(self.currency, self.tf, self.ws))







class TestList:
    """
    This creates the list of tar files that it should search for along with the path name

    #how to use for later
    for i in file_list:
        print(file_list[i]['path'])
    """
    def __init__(self, membership, ws):
        self.md = membership_dict[membership]
        self.ws = ws 
        min_ws = round_number(self.ws - self.md['ws_range'])
        max_ws = round_number(self.ws + self.md['ws_range'])
        xx = np.arange(min_ws, max_ws, 10)

        #update path name when you have it
        file_list = {}
        for curr in self.md['currency']:
            for tf in self.md['tf']:
                for ws in xx:
                    filename = FileName(curr, tf, ws)
                    file_list[filename.name] = {'path': 'idk', 'topk': {}}
        self.file_list = file_list




class ImgMatch:
    """
    """
    def __init__(self, snip_img , model: None, membership: str) -> None:
        self.snip_img = snip_img

        if model == 'CNN':
           self.model = cnn  
        elif model == 'VAE':
            self.model = vae  
        else:
            raise Exception('select model') 
        

    def train_loop(self, epochs):


        self.model.to(self.device)  
        self.model.train()  

        for epoch in range(epochs+1):
            running_loss = 0
            for step, x in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x = x[0].view(-1, 9216).to(self.device)
                targets = (x > 0.5).float().to(self.device) #used for loss_crit for some reason
                y_hat, logvar, mu, z = self.model(x)
                loss = loss_criterion(y_hat, targets, logvar, mu)
                loss.backward()
                self.optimizer.step()
                running_loss += loss





