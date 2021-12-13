from io import BytesIO
from PIL import Image
import base64
import numpy as np
import matplotlib.pyplot as plt
import math

import torch
from torch.optim import Adam  
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import webdataset as wds
from torch.utils.data import TensorDataset

from models.vae import Vae 
# from models.cnn import Cnn
from models.loss import recon_loss, loss_criterion
from utils import img_sizes, membership_dict, round_number










def rolling_images(imgs, path, meta):
    sink = wds.TarWriter(path)
    for index, i in enumerate(imgs):
        x = handle_images(imgs[index])
        ii = torch.Tensor(x[:,:,3])
        ii = torch.unsqueeze(ii, 0)

        sink.write({
            "__key__": "sample%06d" % index,
            "input.pyd": ii,
            "output.pyd": meta['startDate'][index],
        })
    sink.close()










def handle_images(images):
    x = images.split(',', 1)
    img_bytes = base64.b64decode(x[1]) 
    image_PIL = Image.open(BytesIO(img_bytes))
    image_np = np.array(image_PIL)
    return image_np 




def save_snip(img, path, meta):
    x = img.split(',', 1)
    img_bytes = base64.b64decode(x[1]) 
    image_PIL = Image.open(BytesIO(img_bytes))
    x = np.array(image_PIL)
    sink = wds.TarWriter(path)
    ii = torch.Tensor(x[:,:,3])
    ii = torch.unsqueeze(ii, 0)
    sink.write({
        "__key__": "sample%06d",
        "input.pyd": ii,
        "output.pyd": meta['startDate'],
    })
    sink.close()








class ScaleImg:
    def __init__(self, img):
        self.pre_img = img
        self.fix_image()
        self.get_scale()
        self.scale_img()


    def fix_image(self):
        x = self.pre_img.split(',', 1)
        img_bytes = base64.b64decode(x[1]) 
        image_PIL = Image.open(BytesIO(img_bytes))
        np_img = np.array(image_PIL) #has (x, x, 4).shape
        ii = torch.Tensor(np_img[:, :, 3])
        self.fixed_img = torch.unsqueeze(ii, 0) #fixed (1, x, x)


    def get_scale(self):
        self.avg = math.ceil((len(self.fixed_img[0][0]) + len(self.fixed_img[0][:,1])) / 2)
        if self.avg <= img_sizes['small']['dim']:
            self.size = img_sizes['small']
        elif self.avg <= img_sizes['med']['dim']:
            self.size = img_sizes['med']
        else:
            self.size = img_sizes['large']


    def scale_img(self):
        x = self.size['dim']
        resize_img = transforms.Resize(size=(x, x))
        self.img = resize_img(self.fixed_img) 










class FileName:
    """
    #standardizes filename and pathname
    """
    
    def __init__(self, currency, tf, ws):
        self.currency = currency
        self.tf = tf 
        self.ws = ws 
        
        self.file = self.__str__()
        self.path = '/home/pezdel/vsCode/server/tars/{}.tar'.format(self.__str__())

    def __str__(self): #FILENAME
        return('{}_{}_{}'.format(self.currency, self.tf, self.ws))








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
                    file_list[filename.file] = {'path': filename.path, 'topk': {}}
        self.file_list = file_list 









class ImgMatch:
    """
    """

    def __init__(self, train_img, test_list, model, size):
        self.train_img = train_img
        self.test_list =test_list
        self.size = size
        self.device = 'cuda'
        #self.train_dataset = TensorDataset(self.train_img)
        self.train_loader = DataLoader(dataset=train_img)

        for i in train_img:
            for ii in i:
                self.h = len(ii[0][:,1])
                self.w = len(ii[0][0]) 
        print(self.h)
        print(self.w)
        self.input = self.h*self.w
        self.hidden = math.ceil(self.input/3)


        if model == 'CNN':
            pass
           # self.model = Cnn(self.size['input'], self.size['hidden'])  
        elif model == 'VAE':
            self.model = Vae(self.input, self.hidden)  
            # self.model = Vae(self.size['input'], self.size['hidden'])  
        else:
            raise Exception('select model') 
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
                # x = x[0][0].view(-1, self.size['input']).to(self.device)
                targets = (x > 0.5).float().to(self.device) 
                y_hat, logvar, mu, z = self.model(x)
                loss = loss_criterion(y_hat, targets, logvar, mu)
                loss.backward()
                self.optimizer.step()
                running_loss += loss





   
   

    def test_loop(self):
        dist = [] 
        self.model.eval()  

        for i in self.test_list:
            path = self.test_list[i]['path']
            print(path)
            preproc = transforms.Compose([transforms.Resize((self.h, self.w))])
            self.test_dataset = (wds.WebDataset(path).decode('torch').to_tuple('input.pyd').map_tuple(preproc))
            self.test_loader = DataLoader(dataset = self.test_dataset)
            break


        with torch.no_grad():
            for step, x in enumerate(self.test_loader):
                x = x[0][0][0].view(-1, self.input).to(self.device)
                y_hat, logvar, mu, z = self.model(x)
                targets = (x > 0.5).float().to(self.device) #used for loss_crit for some reason
                loss = loss_criterion(y_hat, targets, logvar, mu)

                if not dist:
                    dist = [loss.item()]
                else:
                    dist.append(loss.item())

        dist = np.array(dist)
        self.topk_arr = self.__topk(dist)
        print(self.topk_arr)
        return self.topk_arr
        

    def testing(self):
        plt.imshow(self.train_img[0])
        plt.show()

        # print(self.topk_arr[1][0].item())

        for index, i in enumerate(self.test_dataset):
            for ii in i:
                if index == self.topk_arr[1][0].item():
                    plt.imshow(ii[0])
                    plt.show()











