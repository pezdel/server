from flask import Flask, jsonify, request, Response
from flask_cors import CORS
# import pandas as pd 
from arctic import Arctic

from main import ScaleImg, TestList, ImgMatch, FileName
from main import rolling_images, save_snip
from utils import read_from_arctic
arcticDB = Arctic('mongodb+srv://dbUser:chilicki89@cluster0.lubnq.mongodb.net/testOne?retryWrites=true&w=majority')
import webdataset as wds



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

        # path = '/home/pezdel/vsCode/server/tars/sample_snip/One.tar' 
        # save_snip(img, path, meta)

        scale = ScaleImg(img)  
        test_list = TestList(membership, meta['ws'])
        x = ImgMatch(scale.fixed_img, test_list.file_list, 'VAE', scale.size)
        x.train_loop(50)
        x.test_loop()
        x.testing()
    return jsonify({"content":"nothing!"})
 












#TODO


@app.route('/create_tar', methods=['POST'])
def tar():
    images = request.json['data']
    meta = request.json['meta']
    names = FileName(meta['currency'], meta['tf'], meta['ws'])
    rolling_images(images, names.path, meta)

    return jsonify({"content":"nothing!"})








