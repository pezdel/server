from flask import Flask, jsonify, request, Response
from flask_cors import CORS
# import pandas as pd 
from arctic import Arctic

from main import ScaleImg, TestList, ImgMatch, FileName
from utils import read_from_arctic
arcticDB = Arctic('mongodb+srv://dbUser:chilicki89@cluster0.lubnq.mongodb.net/testOne?retryWrites=true&w=majority')



app = Flask(__name__)
CORS(app)




@app.route('/get_data', methods=['GET'])
def data():
    df = read_from_arctic('EURUSD', '1d')
    return jsonify(data=df.data.to_dict(orient='records'), meta='EURUSD')





#TODO: make this work for currency/tf changes, Redis
@app.route('/get_api', methods=['POST'])
def api():
    pass





#TODO: Redis? 
@app.route('/snip_data', methods=['POST'])
def snip():
    if request.json != None:
        img = request.json['img']
        meta = request.json['meta']
        membership = request.json['membership']

        scale = ScaleImg(img)  
        test_list = TestList(membership, meta['ws'])
        print("here")
        x = ImgMatch(scale.img, test_list.file_list, 'VAE', scale.size)
        x.train_loop(500)
        x.test_loop()


        # x = ImgMatch(np_img, options['model'], options['topx'], options['membership'])
    return jsonify({"content":"nothing!"})
 




#so want to take these and put them in the right folder with right name that we can access later
#if we give the same curr, tf and ws?
#

@app.route('/create_tar', methods=['POST'])
def tar():
    images = request.json['data']
    meta = request.json['meta']
    filename = FileName(meta['currency'], meta['tf'], meta['ws'])
    rolling_images(images, filename, meta)

    return jsonify({"content":"nothing!"})



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






