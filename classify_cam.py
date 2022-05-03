import sys
import os

import flask
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from re import I
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_cors import CORS, cross_origin
from flask import send_from_directory
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Disable tensorflow compilation warnings to skip
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#import tensorflow as tf

def predict(image_data):

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string
    return res, max_score

# Loads label file, removes the carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("logs/trained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


sess = tf.Session()
    # Send the detected image data to the graph to receive the first prediction from trained model
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

def imageRead (random_name):
    c = 0
    global sess
    global softmax_tensor

    res, score = '', 0.0
    i = 0
    mem = ''
    consecutive = 0
    sequence = ''

    while True:
        img = cv2.imread('temp_img/'+random_name)
        img = cv2.flip(img, 1)

        c += 1
        image_data = cv2.imencode('.jpg', img)[1].tostring()
        
        a = cv2.waitKey(1) # check if esc requested
        
        res_tmp, score = predict(image_data)
        res = res_tmp
       
        print(res)
        return res;

@app.route('/image', methods=['GET', 'POST'])
@cross_origin()
def image():
    req = request.get_json()
    random_name = "test" + '.jpg'
    image_data = req['image_data'].split(',')[1]
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    im.save('temp_img/'+random_name, 'JPEG')
    
    imageData = imageRead(random_name)
    return '{"status":1, "value": "'+imageData+'"}';

@app.route('/')
@cross_origin()
def homePage():
    return render_template('index.html')

@app.route("/audio/<path:path>")
def static_dir(path):
    return flask.send_file("templates/audio/" + path)

@app.route('/image-upload', methods=['GET', 'POST'])
@cross_origin()
def imageUpload():
    req = request.get_json()
    random_name = str( random.randint(1, 9999999) )+ '.jpg'
    image_data = req['image_data'].split(',')[1]
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    im.save('temp_img/'+random_name, 'JPEG')
    
    imageData = imageRead(random_name)
    return '{"status":1, "value": "'+imageData+'"}';


if __name__ == '__main__':
    app.run(debug=True)