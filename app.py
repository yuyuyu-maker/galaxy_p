from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import cnn_cs  # 假设你的CNN预测函数在这个文件中
import data_ycl  # 假设你有图像预处理相关的代码
from flask_cors import CORS
# 初始化Flask应用
app = Flask(__name__)

CORS(app)

# 配置上传的文件夹路径
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 允许的文件类型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# 判断文件是否是允许的类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 设置模型路径和设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = cnn_cs.load_model('galaxy_cnn.pth').to(device)  # 加载你的模型


@app.route('/')
def index():
    return render_template('index.html')  # 返回主页


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # 保存上传的文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 调用预测函数
        prediction = cnn_cs.predict_image(model, file_path)

        # 返回预测结果
        return jsonify({'prediction': prediction})
    else:
        return jsonify({'error': 'Invalid file type'})


if __name__ == '__main__':
    # 创建上传目录（如果不存在）
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # 启动Flask应用
    app.run(debug=True)
