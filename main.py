from os import error
from pandas.core.common import count_not_none
from models import cnn, vae #make this work...should be doable



'''
So far 3 classes 
1: ScaleImg -> just scales the img and options for other stuff if needed
2: Tars -> creates a list of Tars to use during Testing based off membership_lvl and ws
3: ImgMatch -> trains off Scaled Img and then tests vs Tars return based off membership
'''

# xx = ScaleImg(img, )
# np_img = handle_images(img)
# fixed_img = torch_snip(np_img)
# plt_snip(fixed_img[0])
#then we scale it?

#so maybe we do xx = ScaleImg(img, ws?)
#then when we need the img we can scale it based off w/e

class ScaleImg:
    def __init__(self, img):
        self.img = img
        self.handle_images()
        self.torch_snip()
        pass


    def handle_images(images):
        x = images.split(',', 1)
        img_bytes = base64.b64decode(x[1]) 
        image_PIL = Image.open(BytesIO(img_bytes))
        image_np = np.array(image_PIL)
        return image_np 


    def torch_snip(img):
        ii = torch.Tensor(img[:, :, 3])
        ii = torch.unsqueeze(ii, 0)
        return ii



    def scale_snip(img, size):
        resize_img = transforms.Resize(size=(90, 90))
        return resize_img(img) 


    def plt_snip(img):
        plt.imshow(img)
        plt.show()




class Tars:
    def __init__(self, membership, windowsize):
        self.membership = 'Basic'
        self.windowsize = 100


 


class ImgMatch:
    """
    """
    def __init__(self, snip_img: Numpy | Tensor, model: None, membership: str) -> None:
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




   
