import matplotlib.pyplot as plt
import math
import os.path
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset
from torch.optim import Adam  
import torchvision.transforms as transforms
import webdataset as wds

from models.vae import Vae 
from models.loss import loss_criterion



#1:
    #patcher draw img working and moving in sync
#2:
    #canvas creator



    #so how to start? ideally we just do this in a sub file but we would nee
    #need alot of stuff, imgs, a magnify, and a slider that




class PathName:
    def __init__(self, currency, tf):
        self.path = '/home/pezdel/vsCode/server/tars/{}/{}/'.format(currency, tf)

    def __str__(self):
        return self.path


class FileName(PathName):
    def __init__(self, currency, tf, ws):
        super().__init__(currency, tf)
        self.currency = currency
        self.tf = tf 
        self.ws = ws 
        self.file = self.__str__()
        self.path = ('{}{}.tar'.format(self.path, self.__str__()))

    def __str__(self): #FILENAME
        return '{}_{}_{}'.format(self.currency, self.tf, self.ws)









class ImgMatch:
    """
    """
    def __init__(self, snip_img, test_list, model):
        self.test_list = test_list
        self.device = 'cuda'
        self.topk_num = 1 
        self.train_dataset = snip_img
        self.train_loader = DataLoader(dataset=snip_img)
            


        for i in self.train_dataset:
            for ii in i:
                self.h = len(ii[0][:,1])
                self.w = len(ii[0][0]) 

        self.input = self.h*self.w
        self.hidden = math.ceil(self.input/3)



        if model == 'CNN':
            pass #TODO
        elif model == 'VAE':
            self.model = Vae(self.input, self.hidden)  
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.00001)





    def __topk(self, dist_list):
        self.dist_list = torch.from_numpy(dist_list)
        topk_tensor = torch.sort(self.dist_list)
        return topk_tensor





    def train_loop(self, epochs):
        self.model.to(self.device)  
        self.model.train()  

        for epoch in range(epochs+1):
            running_loss = 0
            for step, x in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x = x[0][0].view(-1, self.input).to(self.device)
                targets = (x > 0.5).float().to(self.device) 
                y_hat, logvar, mu, z = self.model(x)
                loss = loss_criterion(y_hat, targets, logvar, mu)
                loss.backward()
                self.optimizer.step()
                running_loss += loss


   





    #TODO BATCH SIZES?
    def test_loop(self):
        self.model.eval()  

        for i in self.test_list:
            path = self.test_list[i]['path']
            preproc = transforms.Compose([transforms.Resize((self.h, self.w))])
            ii = os.path.exists(path)
            if ii:
                print('FOUND {}'.format(path))
                self.test_dataset = (wds.WebDataset(path).decode('torch').to_tuple('input.pyd').map_tuple(preproc))
                self.test_loader = DataLoader(dataset = self.test_dataset)
                topk = self.testing_loop()
                self.test_list[i]['topk'] = topk
            else:
                print("NOT FOUND {}".format(path))



    
    def testing_loop(self):
        dist = [] 
        with torch.no_grad():
            for step, x in enumerate(self.test_loader):
                x = x[0][0].view(-1, self.input).to(self.device)
                y_hat, logvar, mu, z = self.model(x)
                targets = (x > 0.5).float().to(self.device) #used for loss_crit for some reason
                loss = loss_criterion(y_hat, targets, logvar, mu)

                if not dist:
                    dist = [loss.item()]
                else:
                    dist.append(loss.item())

        dist = np.array(dist)
        topk = self.__topk(dist)
        sub_topk =  [topk[1][x].item() for x in range(self.topk_num)]
        return sub_topk 
 





   









# class ScaleSnip:
#     """
#     Fixes snip then scales...not in use
#     """
#     def __init__(self, img):
#         self.fixed_snip = fix_snip(img)
#         self.get_scale()
#         self.scale_img()


#     def get_scale(self):
#         self.avg = math.ceil((len(self.fixed_snip[0][0]) + len(self.fixed_snip[0][:,1])) / 2)
#         if self.avg <= img_sizes['small']['dim']:
#             self.size = img_sizes['small']
#         elif self.avg <= img_sizes['med']['dim']:
#             self.size = img_sizes['med']
#         else:
#             self.size = img_sizes['large']


#     def scale_img(self):
#         x = self.size['dim']
#         resize_img = transforms.Resize(size=(x, x))
#         self.scaled = resize_img(self.fixed_snip) 



# class TestList:

#     def __init__(self, membership, ws):
#         self.md = membership_dict[membership]
#         self.ws = ws 
#         min_ws = round_number(self.ws - self.md['ws_range'])
#         max_ws = round_number(self.ws + self.md['ws_range'])
#         xx = np.arange(min_ws, max_ws, 10)

#         file_list = {}
#         for curr in self.md['currency']:
#             for tf in self.md['tf']:
#                 for ws in xx:
#                     filename = FileName(curr, tf, ws)
#                     file_list[filename] = {'path': filename.path, 'topk': {}}
#         self.file_list = file_list 
    
#     def __str__(self):
#         return self.file_list


