from flask import Flask,jsonify,request
from flask_cors import CORS, cross_origin
import detect_algo
import io, base64
from PIL import Image
import detect_algo
import numpy as np
import cv2
from flask import jsonify

app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/sendimg', methods=['GET', 'POST'])
@cross_origin()
def default_param():
    result = request.json
    a = result["imgEncode"].split(",")
    decoded_data = base64.b64decode(a[1])
    np_data = np.fromstring(decoded_data,np.uint8)
    img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    a = detect_algo.import_img(img)   
   
    
    return jsonify(a)


if __name__ == '__main__':
    app.run(debug=True)

#"frontend: " + a + " Backend (10)" + ba '''