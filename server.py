from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import pandas as pd 
from arctic import Arctic

from utils_all import rolling_images, read_from_arctic
from utils import ScaleImg, Tars, ImgMatch
from utils import scale_size, membership_dict 

arcticDB = Arctic('mongodb+srv://dbUser:chilicki89@cluster0.lubnq.mongodb.net/testOne?retryWrites=true&w=majority')


app = Flask(__name__)
CORS(app)

'''
1: on_load - when user first loads the page it needs to pull the current eurusd chart data
2: on_change - on curr or tf changes it needs to get the correct data 
3: submit - after they snip a level it sends back {img, meta and membership}
'''

@app.route('/get_data', methods=['GET'])
def data():
    df = read_from_arctic('EURUSD', '1d')
    return jsonify(data=df.data.to_dict(orient='records'), meta='EURUSD')


#TODO: make this work for currency/tf changes
@app.route('/get_api', methods=['POST'])
def api():
    #redis->then return correct array?
    pass



#TODO: Redis? 
@app.route('/snip_data', methods=['POST'])
def snip():
    if request.json != None:
        img = request.json['img']
        meta = request.json['meta']
        membership = request.json['membership']

        mm = membership_dict[membership]
        scale = ScaleImg(img, meta.ws)  
        tar_list = Tars(membership, meta.ws) #pulls tar files based off ws +- some amount
        loop = ImgMatch(scale, tar_list, membership, model) #takes those params and trains the img,
            #then tests vs the tar_list 

        loop.topk


        #need a way to figure out which size to pick
        

        # x = ImgMatch(np_img, options['model'], options['topx'], options['membership'])


    return jsonify({"content":"nothing!"})
 

