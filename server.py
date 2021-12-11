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

        # scale = ScaleImg(img)  
        test_list = TestList(membership, meta['ws'])


        # loop = ImgMatch(scale, tar_list, membership, model) #takes those params and trains the img,


        # x = ImgMatch(np_img, options['model'], options['topx'], options['membership'])
    return jsonify({"content":"nothing!"})
 

