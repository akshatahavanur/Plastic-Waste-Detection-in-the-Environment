from flask import Flask, render_template, request
import cv2
import torch
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn, retinanet_resnet50_fpn

app = Flask(__name__)

def load_model(model_name):
    if model_name == 'fasterrcnn_resnet50_fpn':
        return fasterrcnn_resnet50_fpn(pretrained=True)
    elif model_name == 'maskrcnn_resnet50_fpn':
        return maskrcnn_resnet50_fpn(pretrained=True)
    elif model_name == 'retinanet_resnet50_fpn':
        return retinanet_resnet50_fpn(pretrained=True)

def preprocess_image(image_path, max_width=800, max_height=600):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image.shape
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))

    image_tensor = ToTensor()(resized_image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor, resized_image

def detect_objects(image_tensor, model):
    with torch.no_grad():
        detections = model(image_tensor)
    return detections

def visualize_detections(image, detections):
    for box in detections[0]['boxes']:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    file = request.files['fileInput']
    file_path = 'static/' + file.filename
    file.save(file_path)

    model_name = request.form['model']
    model = load_model(model_name)
    model.eval()

    image_tensor, resized_image = preprocess_image(file_path, max_width=800, max_height=600)

    detections = detect_objects(image_tensor, model)
    visualize_detections(resized_image, detections)

    output_image_path = 'static/output.jpg'
    cv2.imwrite(output_image_path, resized_image)

    return render_template('result.html', image_path=output_image_path)

if __name__ == '__main__':
    app.run(debug=True)
