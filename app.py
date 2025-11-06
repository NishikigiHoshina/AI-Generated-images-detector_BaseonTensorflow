from flask import Flask, render_template, request, redirect, url_for, send_from_directory,jsonify
import joblib
from preprocessing.text_preprocessor import preprocess_text
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
# 设置上传文件的保存路径
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 确保上传文件夹和静态文件夹存在
for folder in [app.config['UPLOAD_FOLDER'], 'static']:
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        except OSError as e:
            print(f"Error creating folder {folder}: {e}")


# 视频检测函数
def detect_video(file_path):
    # 使用 OpenCV 打开视频
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return "Error opening video file"

    # 读取视频的前几帧进行简单分析
    frame_count = 0
    while frame_count < 10:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 这里可以添加更复杂的检测逻辑
        # 例如：分析帧的直方图、边缘特征等
        # 这里只是一个简单的示例，直接返回一个固定结果
        cap.release()
        return "AI Generated" if frame_count < 5 else "Real Video"

    cap.release()
    return "Real Video"

def SimpleVideoCheck():
    # # 加载预训练模型
    # # 假设我们使用一个简单的预训练模型，这里使用一个简单的卷积神经网络
    # class SimpleCNN(torch.nn.Module):
    #     def __init__(self):
    #         super(SimpleCNN, self).__init__()
    #         self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
    #         self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    #         self.fc1 = torch.nn.Linear(32 * 64 * 64, 128)
    #         self.fc2 = torch.nn.Linear(128, 2)  # 2 类别：AI 生成或真实视频

    #     def forward(self, x):
    #         x = torch.relu(self.conv1(x))
    #         x = torch.max_pool2d(x, 2)
    #         x = torch.relu(self.conv2(x))
    #         x = torch.max_pool2d(x, 2)
    #         x = x.view(x.size(0), -1)
    #         x = torch.relu(self.fc1(x))
    #         x = self.fc2(x)
    #         return x

    # # 加载模型
    # model = SimpleCNN()
    # model.load_state_dict(torch.load('pretrained_model.pth', map_location=torch.device('cpu')))
    # model.eval()

    # # 数据预处理
    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # # 视频检测函数
    # def detect_video(file_path):
    #     cap = cv2.VideoCapture(file_path)
    #     if not cap.isOpened():
    #         return "Error opening video file"

    #     frame_count = 0
    #     predictions = []

    #     while frame_count < 10:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         frame_count += 1

    #         # 将帧转换为 PIL 图像
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         frame = Image.fromarray(frame)

    #         # 数据预处理
    #         frame = transform(frame).unsqueeze(0)

    #         # 模型预测
    #         with torch.no_grad():
    #             output = model(frame)
    #             _, predicted = torch.max(output, 1)
    #             predictions.append(predicted.item())

    #     cap.release()

    #     # 统计预测结果
    #     ai_count = predictions.count(0)
    #     real_count = predictions.count(1)

    #     return "AI Generated" if ai_count > real_count else "Real Video"
    return False




# 加载文本模型和向量器
try:
    text_model = joblib.load('models/fake_news_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    print("Text model loaded successfully")
except Exception as e:
    print(f"Text model loading error: {e}")
    text_model = None
    vectorizer = None

# 加载图片模型
try:
    image_model = load_model('models/image_fake_news_model.h5')
    print("Image model loaded successfully")
except Exception as e:
    image_model = None
    print(f"Image model loading error: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path, target_size=(224, 224)):
    """预处理上传的图片"""
    try:
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # 归一化
        return img_array
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None

def predict_image(image_path):
    """使用图片模型进行预测"""
    if image_model is None:
        return "图片模型未加载"
    
    img_array = preprocess_image(image_path)
    if img_array is None:
        return "图片处理失败"
    
    try:
        prediction = image_model.predict(img_array)
        # 假设模型输出为[fake_probability, real_probability]
        if prediction[0][0] > 0.5:
            return "Fake"
        else:
            return "Real"
    except Exception as e:
        print(f"Image prediction error: {e}")
        return "预测过程出错"



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/videopredict')
def videopredict():
    return render_template('videopredict.html')

@app.route('/predicttext',methods=['POST'])
def predicttext():
    news_text = request.form.get('news_text', '')
    prediction = None
    error_msg = None
    # 处理文本预测
    if text_model and vectorizer and news_text:
        try:
            preprocessed_text = preprocess_text(news_text)
            features = vectorizer.transform([preprocessed_text])
            prediction = text_model.predict(features)[0]
        except Exception as e:
            error_msg = f"文本预测错误: {e}"
            print(error_msg)

    return render_template('index.html', 
                           prediction=prediction, 
                           news_text=news_text,
                           error_msg=error_msg)

@app.route('/predictimg', methods=['POST'])
def predictimg():
    image_file = request.files.get('image_file', None)
    
    image_prediction = None
    image_path = None
    error_msg = None
    
    
    # 处理图片预测
    if image_file:
        if not allowed_file(image_file.filename):
            error_msg = "不支持的图片格式，请上传png/jpg/jpeg/gif"
        else:
            try:
                filename = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
                print(f"准备保存图片到: {filename}")
                image_file.save(filename)
                print(f"图片保存成功: {filename}")
                image_path = url_for('uploaded_file', filename=image_file.filename)
                
                if image_model:
                    image_prediction = predict_image(filename)
                else:
                    image_prediction = "图片模型未加载，无法预测"
            except Exception as e:
                error_msg = f"图片处理错误: {e}"
                print(error_msg)
    
    return render_template('index.html', 
                           image_prediction=image_prediction,
                           image_path=image_path,
                           error_msg=error_msg)

#上传缓存图片的路由方法
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        print(f"Error sending uploaded file: {e}")
        return "文件发送出错", 500

@app.route('/static/<path:filename>')
def static_files(filename):
    try:
        return send_from_directory('static', filename)
    except Exception as e:
        print(f"Error sending static file: {e}")
        return "文件发送出错", 500
    

#上传缓存视频的路由方法
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 保存上传的视频文件
    file_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(file_path)

    try:
        # 调用检测函数
        result = detect_video(file_path)
        return jsonify({'result': result})
    finally:
        # 删除缓存的视频文件
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == '__main__':
    app.run(debug=True, port=5000)