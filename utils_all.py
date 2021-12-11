from io import BytesIO
from PIL import Image
import base64
import numpy as np
import torch
import webdataset as wds


#---------------------------------
#-------helpful things to remember
# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(threshold=np.inf)
# ii = torch.permute(ii, (2, 0, 1))
#-------helpful things to remember
#---------------------------------



#so this file has functions that idk if they will make it 
#to the real server or not


import os
def tars_list():
    path = '/home/pezdel/vsCode/server/src/backend/tars/'
    dir_list = os.listdir(path)
    tar_list = {}

    for tar in dir_list:
        if tar.endswith('.tar'):
            tar_list[tar] = None
    return tar_list





def handle_images(images):
    x = images.split(',', 1)
    img_bytes = base64.b64decode(x[1]) 
    image_PIL = Image.open(BytesIO(img_bytes))
    image_np = np.array(image_PIL)
    return image_np 



#todo:
    #so now we do the full loop




def snip_images(imgs):
    x = handle_images(imgs)
    ii = torch.Tensor(x[:, :, 3])
    ii = torch.unsqueeze(ii, 0)
    sink = wds.TarWriter('single.tar')
    sink.write({
            "__key__": "sample001" ,
            "input.pyd": ii,
        })
    sink.close()
    pass



def rolling_images(imgs, filename, meta):
    sink = wds.TarWriter('/home/pezdel/vsCode/server/src/backend/tars/{}.tar'.format(filename))
    for index, i in enumerate(imgs):
        x = handle_images(imgs[index])
        ii = torch.Tensor(x[:,:,3])
        ii = torch.unsqueeze(ii, 0)

        sink.write({
            "__key__": "sample%06d" % index,
            "input.pyd": ii,
            "output.pyd": meta[index],
        })
    sink.close()



from arctic import Arctic
arcticDB = Arctic('mongodb+srv://dbUser:chilicki89@cluster0.lubnq.mongodb.net/testOne?retryWrites=true&w=majority')
def read_from_arctic(parent, child):
    subDB = arcticDB[parent]
    item = subDB.read(child)
    return item 


