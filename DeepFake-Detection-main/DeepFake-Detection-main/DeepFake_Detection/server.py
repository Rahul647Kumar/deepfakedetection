from flask import Flask, render_template, request, json
from werkzeug.utils import secure_filename
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
import face_recognition
from torch import nn
import warnings

warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'Uploaded_Files'
app = Flask("__main__", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Creating Model Architecture
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()

        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])

        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)

        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


# Image normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

sm = nn.Softmax(dim=1)

# Dataset class
class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        first_frame = np.random.randint(0, int(100 / self.count))

        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            if faces:  # Check if any face is detected
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]

            frames.append(self.transform(frame))

            if len(frames) == self.count:
                break

        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image


# Prediction function
def predict(model, img, device):
    img = img.to(device)
    fmap, logits = model(img)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    print('Confidence of prediction:', confidence)
    return [int(prediction.item()), confidence]


# Video detection function
def detect_fake_video(video_path):
    im_size = 112
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load video and model
    path_to_videos = [video_path]
    video_dataset = ValidationDataset(path_to_videos, sequence_length=20, transform=train_transforms)

    model = Model(2).to(device)
    path_to_model = 'model/df_model.pt'
    model.load_state_dict(torch.load(path_to_model, map_location=device))
    model.eval()

    # Prediction
    for i in range(len(path_to_videos)):
        with torch.no_grad():
            prediction = predict(model, video_dataset[i], device)

        if prediction[0] == 1:
            print("REAL")
        else:
            print("FAKE")

    return prediction


# Flask routes
@app.route('/', methods=['POST', 'GET'])
def homepage():
    return render_template('index.html')


@app.route('/Detect', methods=['POST', 'GET'])
def detect_page():
    if request.method == 'POST':
        video = request.files['video']
        video_filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)

        prediction = detect_fake_video(video_path)
        output = "REAL" if prediction[0] == 1 else "FAKE"
        confidence = prediction[1]
        data = json.dumps({'output': output, 'confidence': confidence})

        os.remove(video_path)
        return render_template('index.html', data=data)

    return render_template('index.html')


app.run(port=3000)
