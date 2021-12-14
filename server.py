from flask import Flask, jsonify, request, Response
from flask_cors import CORS
# import pandas as pd 
from arctic import Arctic

from main import ImgMatch, FileName
from utils import fix_snip, create_test_list, read_from_arctic, save_snip, rolling_images
import webdataset as wds



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
@app.route('/snip_submit', methods=['POST'])
def snip():
    if request.json != None:
        img = request.json['img']
        meta = request.json['meta']
        membership = request.json['membership']

        snip = fix_snip(img)
        test_list = create_test_list(membership, meta['ws'])

        x = ImgMatch(snip, test_list, 'VAE')
        x.train_loop(500)
        x.test_loop()
        # x.testing()
    return jsonify({"content":"nothing!"})
 






@app.route('/save_snip', methods=['POST'])
def save():
    if request.json != None:
        img = request.json['img']
        meta = request.json['meta']
        membership = request.json['membership']
        path = '/home/pezdel/vsCode/server/tars/sample_snip/One.tar' 
        save_snip(img, path, meta)
    return jsonify({"content":"nothing!"})










@app.route('/create_tar', methods=['POST'])
def tar():
    if request.json != None:
        images = request.json['data']
        meta = request.json['meta']
        names = FileName(meta['currency'], meta['tf'], meta['ws'])
        # rolling_images(images, names.path, meta)

        return jsonify({"content":"nothing!"})








