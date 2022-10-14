from flask import Flask, request, render_template
# from jojo_main import predict
import sys
sys.path.append(".")
sys.path.append("..")
from smtgan_predict import flask_inference
from train import flask_train
from werkzeug.utils import secure_filename
# from PIL import Image
# import torchvision.transforms as transforms
import os
# import progress_bar

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
        # img = Image.open(f.stream)
        # transform = transforms.Compose([transforms.ToTensor()])
        # img = transform(img)
        # f.save(secure_filename(f.filename))

        # input or reference 의 컬러 정보 유지할 것인지를 사용자가 선택
        # preserve_color = request.form.get('preserve_color')
        save_name = request.form.get('save_name')
        # 학습 진행
        # jojogan_train(f.filename, preserve_color)
        jojogan_train(save_name, f.filename)
        # jojogan_train(f.filename, preserve_color)
        # render_template('train.html', value=idx)

        # return "img"
        # return render_template('train.html')
        return render_template('main.html')
    return render_template('train.html')

# def train_percent(value):
#     return render_template('train.html', value=value)

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
        print('select model name:', model_name)
        f.save('demo_flask/static/face_input_image/{0}'.format(secure_filename(f.filename)))  # 얼굴 이미지 저장.

        # input or reference 의 컬러 정보 유지할 것인지를 사용자가 선택
        preserve_color_info = request.form.get('preserve_color')
        jojogan_inference(f.filename, model_name, preserve_color_info)
        # filename = '/home/project/Model-Deployment-Using-Flask/input_image/{0}'.format(secure_filename(f.filename))
        # filename1 = 'input_image.{0}'.format(f.filename)
        # print(f'{f.filename.split(".")[0]}.png')

        name = f.filename.split(".")[0]
        model_name = model_name.split(".")[0]
        return render_template("inference_result.html",
                               toonify=f'{model_name}/{name}_result.png')
                               # disgust=f'{model_name}/disgust/disgust.gif',
                               # happy=f'{model_name}/happy/happy.gif',
                               # surprise=f'{model_name}/surprise/surprise.gif')

# models function
def jojogan_train(save_name, img_name):
    # refer_img, img_name = get_refer_img()

    # for idx in flask_train(save_name, img_name):
    #     print(idx)

    flask_train(save_name, img_name)

    # return Response(generate(), mimetype='text/event-stream')
    # Response(progress_bar.progress(data for data in train_iter), mimetype='text/event-stream')
    # print('print ', a)
    # return Response(progress_bar.progress(flask_train(img_name, preserve_color)), mimetype='text/event-stream')

def jojogan_inference(img_name, model_name, preserve_color_info='refer'):
    # face_img = get_face_img()
    flask_inference(img_name, model_name, preserve_color_info)
    # return toonify


# if __name__ == '__main__':
    # app.run(host='0.0.0.0', port='8080', debug=True)
    # app.run(host='0.0.0.0')

if __name__ == '__main__':

    # if __package__ is None:
    #     import sys
    #     from os import path
    #
    #     print(path.dirname(path.dirname(path.abspath(__file__))))
    #     sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    #     app.run(host='0.0.0.0')

    # app.run(host='0.0.0.0', debug=True)
    # app.run(host="0.0.0.0", debug=False)
    print('test')
    app.run()