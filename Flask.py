from flask import Flask, request, jsonify
import cnn_cs
import os
import torch
app = Flask(__name__)

# 加载模型
model_path = "galaxy_cnn.pth"
model = cnn_cs.load_model(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


@app.route('/predict', methods=['POST'])
def predict_api():
    # 获取上传的文件
    img_file = request.files['image']
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    # 创建目录，如果目录已存在则不会报错
    # 保存文件到临时路径
    img_path = os.path.join('uploads', img_file.filename)
    img_file.save(img_path)

    # 调用预测函数
    prediction = cnn_cs.predict_image(model, img_path)

    return jsonify({'prediction': prediction})


if __name__ == "__main__":
    app.run(debug=True)