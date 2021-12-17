import webdataset as wds
import torch
from io import BytesIO
from PIL import Image
import base64
import numpy as np
from arctic import Arctic
from main import FileName










def fix_snip(img):
    """
    vip
    """
    x = img.split(',', 1)
    img_bytes = base64.b64decode(x[1]) 
    image_PIL = Image.open(BytesIO(img_bytes))
    np_img = np.array(image_PIL) #has (x, x, 4).shape
    ii = torch.Tensor(np_img[:, :, 3])
    fixed_snip = torch.unsqueeze(ii, 0) #fixed (1, x, x)
    return fixed_snip










def create_test_list(membership, ws):
    """
    vip
    """
    md = membership_dict[membership]
    ws = ws 
    min_ws = round_number(ws - md['ws_range'])
    max_ws = round_number(ws + md['ws_range'])
    xx = np.arange(min_ws, max_ws, 10)

    file_list = {}
    for curr in md['currency']:
        for tf in md['tf']:
            for ws in xx:
                filename = FileName(curr, tf, ws)
                file_list[filename.file] = {'path': filename.path, 'topk': {}}
    return file_list










def scale_snip(img):
    """
    unused
    """
    fixed_snip = fix_snip(img)

    avg = math.ceil((len(fixed_snip[0][0]) + len(fixed_snip[0][:,1])) / 2)
    if avg <= img_sizes['small']['dim']:
        size = img_sizes['small']
    elif avg <= img_sizes['med']['dim']:
        size = img_sizes['med']
    else:
        size = img_sizes['large']
    x = size['dim']
    resize_img = transforms.Resize(size=(x, x))
    scaled = resize_img(fixed_snip) 

    return scaled










def rolling_images(imgs, path, meta):
    '''
    used when created tar rolling files
    '''
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
    '''
    used when createing tar datafile, simular to other function can probably remove at some point?
    '''
    x = images.split(',', 1)
    img_bytes = base64.b64decode(x[1]) 
    image_PIL = Image.open(BytesIO(img_bytes))
    image_np = np.array(image_PIL)
    return image_np 










def save_snip(img, path, meta):
    '''
    used when wanting to create single tar snip file for testing in JN
    '''
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



    #so clean up on chart is getting big
    #figure out how to organize it better
    #startWindowSlider is the only thing that should kick off updates on price/date 
    #then update mag 
    #fix price and date 
    #standardize snip/mag scale 
    #export snip








arcticDB = Arctic('mongodb+srv://dbUser:chilicki89@cluster0.lubnq.mongodb.net/testOne?retryWrites=true&w=majority')
def read_from_arctic(parent, child):
    subDB = arcticDB[parent]
    item = subDB.read(child)
    item = item.data
    if len(item) >2000:
        print(len(item))
        item = item.iloc[-2000:,:]
    return item 










def round_number(num):
    rounded = round(num/10)*10
    return rounded










#SIZE DICTONARY-------not currently used
small = {'dim': 50,
         'input': 2500,
         'hidden': 1000}
med = {'dim': 100,
       'input': 10000,
       'hidden': 5000}
large = {'dim': 150,
         'input': 22500,
         'hidden': 8000}

img_sizes = {'small': small, 'med': med, 'large': large}





#TODO: make this a class? idk but also make option for vae into cnn for member/elite, basic is just vae
basic_currency = ['EURUSD', 'GBPUSD', 'USDJPY', 'NZDUSD', 'AUDUSD']
basic_tf = ['1d', '1w']
basic = {'currency': basic_currency,
         'tf': basic_tf,
         'ws_range': 10,
         'topx': 1}
member_currency = ['BTCUSD', 'SNP', 'USDX', 'ETHUSD']
member_tf = ['4h', '1h']
member = {'currency': basic_currency + member_currency,
          'tf': basic_tf + member_tf,
          'ws_range': 50,
          'topx': 5}
elite_currency = ['stock', 'stock']
elite_tf = ['5m', '15m']
elite = {'currency': basic_currency + member_currency + elite_currency,
         'tf': basic_tf + member_tf + elite_tf,
         'ws_range': 50,
         'topx': 5}
membership_dict = {'basic': basic, 'member': member, 'elite': elite}


