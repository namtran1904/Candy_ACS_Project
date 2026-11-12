import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import RTDETR
from PIL import Image

#khối 1
DATA_DIR = os.path.join("data", "data_set", "train_data_set")

if(torch.cuda.is_available()):
    device = torch.device ("cuda")
else:
    device = torch.device("cpu")
print(f" Đang chạy trên nền tảng: {device}")

#nạp mô mình detect
detector = RTDETR("rtdetr-l.pt")
print(f"Nạp thành công mô hình Detect (RT-DETR)!")

class_names = sorted(os.listdir(DATA_DIR))
num_classes = len(class_names)
print(f"Tổng cộng có {num_classes} lớp kẹo")
print(class_names)

classifier = models.resnet18(weights=None)

classifier.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(classifier.fc.in_features,num_classes)
)

classifier_path = "candy_resnet18.pth"

if os.path.exists(classifier_path):
    classifier.load_state_dict(torch.load(classifier_path,map_location = device))
    classifier = classifier.to(device)

    classifier.eval()
    print(f"đã nap thành công mô hình Classifier(ResNet18)!")
else:
    print(f"Cânhs báo chưa tìm thấy file {classifier_path}. Hãy kiểm tra lại file đã lưu!")

#khối 2
crop_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
)

def predict_crop(crop_img):
    #đổi định dạng ảnh 
    crop_rgb = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
    pill_img = Image.fromarray(crop_rgb)

    #biến đổi thành tensor
    input_tensor = crop_transforms(pill_img)
    input_full_batch = input_tensor.unsqueeze(0).to(device)


    with torch.no_grad():
        outputs = classifier(input_full_batch)

        #dùng hàm softmax
        possibilities = torch.nn.functional.softmax(outputs[0], dim = 0)

        #xác suất cao nhất và lớp tương ứng
        confidence, predicted = torch.max(possibilities, dim = 0)

        predicted_class = class_names[predicted.item()]

        #lấy % tin cậy
        confidence_score = confidence.item()*100

        return predicted_class, confidence_score

