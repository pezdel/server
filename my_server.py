from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import pandas as pd 
from arctic import Arctic

from utils_all import rolling_images, read_from_arctic
from utils import ScaleImg
from utils import scale_size 
arcticDB = Arctic('mongodb+srv://dbUser:chilicki89@cluster0.lubnq.mongodb.net/testOne?retryWrites=true&w=majority')



app = Flask(__name__)
CORS(app)


#-------------------------------Real server------------------------------------
#on_load
@app.route('/get_data', methods=['GET'])
def data():
    df = read_from_arctic('EURUSD', '1d')
    return jsonify(data=df.data.to_dict(orient='records'), meta='EURUSD')


#on_change
#TODO: make this work for currency/tf changes
@app.route('/get_api', methods=['POST'])
def api():
    #redis->then return correct array?
    pass



#snip submit
#TODO: Redis? scale img to correct size 
@app.route('/snip_data', methods=['POST'])
def snip():
    if request.json != None:
        img = request.json['img']
        meta = request.json['meta']
        membership = request.json['membership']

        scale_size(50)
        #need a way to figure out which size to pick
        # xx = ScaleImg(img, )
        # np_img = handle_images(img)
        # fixed_img = torch_snip(np_img)
        # plt_snip(fixed_img[0])



        # x = ImgMatch(np_img, options['model'], options['topx'], options['membership'])


    return jsonify({"content":"nothing!"})
    

#so frontend sends img: url, meta: {curr, tf, ws}, membership: str

#route-----




#extra routes
#----------------------------------OTHER SERVER STUFF FOR ME--------------------------
#-----------------------------------------------ARCTIC--------------------------------------------------------------
#Update Button
@app.route('/update_db', methods=['POST'])
def one():
    df = request.json
    # dash = update_db(df)
    # return jsonify(dash.to_dict(orient='records')) 



#onLoad to get w/e is loaded in the template spot
@app.route('/get_template', methods=['GET'])
def get_temp():
    df = read_from_arctic('APIs', 'template')
    return jsonify(df.data.to_dict(orient='records'))



@app.route('/get_currency_list', methods=['GET'])
def get_currency():
    db = read_from_arctic('APIs', 'template')
    df = pd.DataFrame(db.data)
    df.iloc[0,0] = 'NZDUSD'
    df = df.drop_duplicates(subset=['currency'])
    arr = [] 
    for index, row in df.iterrows():
        arr.append({'value': row['currency'], 'label': row['currency']})
    arr = pd.DataFrame(arr)

    return jsonify(arr.to_dict(orient='records'))




#----------------------------------------------IMGAGES-------------------------------------------

@app.route('/rolling_data', methods=['POST'])
def rolling():
    images=request.json['data']
    currency = request.json['currency']
    tf = request.json['tf']
    meta = request.json['meta']
    window_size = request.json['windowSize']
    jump = request.json['jump']
    filename = currency + '_' + str(tf) + '_' + str(window_size) + '_' + str(jump)
    rolling_images(images, filename, meta)
    return jsonify({"content":"nothing!"})


if __name__=='__main__':
    app.run()
    pass























