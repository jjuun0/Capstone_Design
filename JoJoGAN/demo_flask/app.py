from flask import Flask, render_template, request, send_file
import io
import json
import base64
from io import BytesIO
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
import pickle
import cv2
import numpy as np
import os

# models = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        f = request.files['file']

        img = Image.open(f.stream)
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
        # f.save(secure_filename(f.filename))
        return img
    # img = Image.open(getimg)
    #
    #
    # json_data = request.get_json()  # Get the POSTed json
    # dict_data = json.loads(json_data)  # Convert json to dictionary
    #
    # img = dict_data["image"]  # Take out base64# str
    # img = base64.b64decode(img)  # Convert image data converted to base64 to original binary data# bytes
    # img = BytesIO(img)  # _io.Converted to be handled by BytesIO pillow
    # img = Image.open(img)
    # img_shape = img.size  # Appropriately process the acquired image
    # name = 1
    # text = dict_data["text"] + "fuga"  # Properly process with the acquired text
    # img.save('./static/'+name+'.png', 'PNG')
    # # Return the processing result to the client
    # return render_template('after.html', data=name)
    return 'file saved'

@app.route('/model_test', methods=['GET', 'POST'])
def set_inference():
    if request.method =='GET':
        file_list = os.listdir('demo_flask/static/models')
        return render_template('inference.html', file_list=file_list)

    # inference face image
    if request.method == 'POST':
        f = request.files['file']
        select = request.form.get('file_list')  # select models's name
        print(select)
        f.save('demo_flask/static/face_input_image/{0}'.format(secure_filename(f.filename)))
        # filename = '/home/project/Model-Deployment-Using-Flask/input_image/{0}'.format(secure_filename(f.filename))
        # filename1 = 'input_image.{0}'.format(f.filename)
        return render_template("index.html", user_image=f.filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
