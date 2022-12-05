import sys
import os
sys.path.append(".")
sys.path.append("..")

from train_transmixing import flask_train
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template


app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')


@app.route('/get', methods=['GET', 'POST'])
def get_refer_img():
    if request.method == 'POST':
        f = request.files['file']
        # 캐릭터 이미지 받아옴.
        f.save(os.path.join('/home/project/바탕화면/Capstone_Design/JoJoGAN/style_images', secure_filename(f.filename)))

        # input or reference 의 컬러 정보 유지할것인지를 사용자가 선택
        preserve_color = request.form.get('preserve_color')

        # 학습 진행
        train(f.filename, preserve_color)
        return render_template('home.html')


@app.route('/model_test', methods=['GET', 'POST'])
# inference 과정때 필요한 얼굴 이미지와, 어떤 모델을 가지고 style transfer를 진행할지 사용자한테 인풋을 받는다.
def set_inference():
    if request.method =='GET':
        file_list = os.listdir('demo_flask/static/models')  # 모델 리스트
        return render_template('inference.html', file_list=file_list)

    # inference face image
    if request.method == 'POST':
        f = request.files['file']
        # select = request.form.get('file_list')  # 모델 선택
        # print(select)
        f.save('demo_flask/static/face_input_image/{0}'.format(secure_filename(f.filename)))  # 얼굴 이미지 저장.
        return render_template("index.html", user_image=f.filename)


# models function
def train(img_name, preserve_color='input'):
    flask_train(img_name, preserve_color)


if __name__ == '__main__':
    app.run(host='0.0.0.0')