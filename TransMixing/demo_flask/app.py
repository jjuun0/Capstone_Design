import sys
import os
sys.path.append(".")
sys.path.append("..")

from predict_transmixing import flask_inference
from train_transmixing import flask_train
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template


app = Flask(__name__)

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/model_train', methods=['GET', 'POST'])
def get_refer_img():
    if request.method == 'POST':
        f = request.files['refer_img']
        # 캐릭터 이미지 받아옴.
        f.save(os.path.join('/home/project/바탕화면/Capstone_Design/JoJoGAN/style_images', secure_filename(f.filename)))

        save_name = request.form.get('save_name')
        # 학습 진행
        jojogan_train(save_name, f.filename)

        return render_template('main.html')
    return render_template('train.html')


@app.route('/model_inference', methods=['GET', 'POST'])
# inference 과정때 필요한 얼굴 이미지와, 어떤 모델을 가지고 toonify 를 진행할지 사용자한테 인풋을 받는다.
def set_inference():
    if request.method == 'GET':
        file_list = os.listdir('demo_flask/static/models')  # 모델 리스트
        return render_template('inference.html', file_list=file_list)

    # inference face image
    if request.method == 'POST':
        f = request.files['face_img']
        model_name = request.form.get('file_list')  # 모델 선택
        # print('select model name:', model_name)
        f.save('demo_flask/static/face_input_image/{0}'.format(secure_filename(f.filename)))  # 얼굴 이미지 저장.

        # input or reference 의 컬러 정보 유지할 것인지를 사용자가 선택
        preserve_color_info = request.form.get('preserve_color')
        jojogan_inference(f.filename, model_name, preserve_color_info)

        name = f.filename.split(".")[0]
        model_name = model_name.split(".")[0]
        return render_template("inference_result.html",
                               toonify=f'{model_name}/{name}_result.png')

# models function
def jojogan_train(save_name, img_name):
    flask_train(save_name, img_name)


def jojogan_inference(img_name, model_name, preserve_color_info='refer'):
    flask_inference(img_name, model_name, preserve_color_info)


if __name__ == '__main__':
    app.run()