from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import tensorflow as tf
import base64
import pandas as pd
import numpy as np
from numpy import asarray
import traceback
from flask import Flask, request
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from flask_restful import Api
import io
from scipy.special import softmax
np.set_printoptions(precision=5)
import json

size = 32

host = '0.0.0.0'
model_name = 'sample_autoencoder_model_13'
port = '7002'
app = Flask('BrickBanker Inference')
api = Api(app)
app.config['JSON_AS_ASCII'] = False


model = tf.keras.models.load_model(f'{model_name}.h5')
encoder = tf.keras.models.load_model("encoder_alldata_3.h5")


@app.route('/', methods=['GET'])
def index():
    return "Welcome to BrickBanker Project!"


# Process image and predict label
def processImg(IMG_PATH):
    test = []
    # load the image
    #image = Image.open(IMG_PATH)
    image = IMG_PATH.resize((size, size))

    # convert image to numpy array
    data = asarray(image)
    test.append(data)
    x_test = np.asanyarray(test)
    x_test = x_test.astype('float32') / 255.

    res = model.predict(x_test)
    return res[0]


def euclidean(a, b):
    # compute and return the euclidean distance between two vectors
    return np.linalg.norm(a - b)


def perform_search(queryFeatures, index, maxResults=5):
    # initialize our list of results
    results = []
    # loop over our index
    for i in range(0, len(index["features"])):
        # compute the euclidean distance between our query features
        # and the features for the current image in our index, then
        # update our results list with a 2-tuple consisting of the
        # computed distance and the index of the image
        d = euclidean(queryFeatures, index["features"][i])
        results.append((d, i))
    # sort the results and grab the top ones
    results = sorted(results)[:maxResults]
    # return the list of results
    return results


index = pickle.loads(open("index_alldata_3.pkl", "rb").read())
x_test = pickle.loads(open("img_all_3.pkl", "rb").read())['img']

def preprocessing(IMG_PATH):
    inp = []
    # load the image
    #image = Image.open(IMG_PATH)
    image = IMG_PATH.resize((size, size))
    # convert image to numpy array
    data = asarray(image)
    inp.append(data)
    query = np.asanyarray(inp)
    query = query.astype('float32') / 255.

    res = encoder.predict(query)

    queryFeatures = res[0]
    results = perform_search(queryFeatures, index, maxResults=5)
    images = []
    part_numbers = []
    # loop over
    for (d, j) in results:
        # grab the result image, convert it back to the range
        # [0, 255, and then update the images list
        image = (x_test[j] * 255).astype("uint8")
        image = np.dstack([image])
        images.append(image)
        part_numbers.append(index['brick_codes'][j])

    plt.figure(figsize=(10, 10))
    for i in range(5):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(part_numbers[i])
        plt.axis("off")
    plt.savefig('similarImages.jpg')
    return 'similarImages.jpg'


def get_part_numbers(IMG_PATH):
    inp = []
    # load the image
    #image = Image.open(IMG_PATH)
    image = IMG_PATH.resize((size, size))

    # convert image to numpy array
    data = asarray(image)
    inp.append(data)
    query = np.asanyarray(inp)
    query = query.astype('float32') / 255.

    res = encoder.predict(query)

    queryFeatures = res[0]
    results = perform_search(queryFeatures, index, maxResults=5)
    images = []
    part_numbers = []
    # loop over
    x = []
    for (i, j) in results:
        x.append((100 - i))
    m = softmax(x)
    i=0
    for (d, j) in results:
        # grab the result image, convert it back to the range
        # [0, 255, and then update the images list
        image = (x_test[j] * 255).astype("uint8")
        image = np.dstack([image])
        images.append(image)
        
        part_numbers.append({"partId": index['brick_codes'][j],
                                 "score": str(m[i]*100)})
        i+=1
    return part_numbers


@app.route("/similarImages", methods=["POST"])
def find_similar():
    img = request.files.get('image').read()
    img = Image.open(io.BytesIO(img))
    #print(img)
    #data.save("image.jpg")
    # inp = request.files["queryImage"]
    result = preprocessing(img)
    #plt.imsave("similarImages.jpg", result)
    with open(result, "rb") as image_file:
        return base64.b64encode(image_file.read())
    return ""


@app.route("/part_numbers", methods=["POST"])
def part_numbers():
    img = request.files.get('image').read()
    img = Image.open(io.BytesIO(img))
    res = get_part_numbers(img)
    return {
        'code': 200,
        'result': res
    }
    #return res.to_json(orient='records')


# Process images
@app.route("/autoencoder", methods=["POST"])
def processReq():
    img = request.files.get('image').read()
    img = Image.open(io.BytesIO(img))
    resp = processImg(img)
    plt.imsave("pred.jpg", resp)
    #plt.imshow("pred.jpg")
    with open("pred.jpg", "rb") as image_file:
        return base64.b64encode(image_file.read())
    return ""

@app.errorhandler(Exception)
def handle_exception(e):
    return {
        'code': 500,
        'error': "Internal Server Error"
    }

def start():
    # app.run(host=host, port=port, ssl_context=('cert.pem', 'key.pem'))
    app.run(host=host,port=port)  # host=host,

if __name__ == '__main__':
    try:
        start()
    except Exception as ex:
        handle_exception()
