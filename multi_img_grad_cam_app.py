import streamlit as st 
from PIL import Image 
import numpy as np 
import torch
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

@st.cache_resource
def load_labels():
    import urllib.request
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = urllib.request.urlopen(url)
    classes = [line.strip().decode('utf-8') for line in response.readlines()]
    return classes


model = load_model()
labels = load_labels()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# UI 구성
st.title('🖼️ 다중 이미지 Grad-CAM 분류기')

uploaded_files = st.file_uploader(
    "이미지를 여러 개 업로드하세요",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=uploaded_file.name, use_column_width=True)

        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output).item()
            class_name = labels[pred_class]

        # Grad-CAM 생성
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
        image_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
        cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

        # 결과 출력
        st.markdown(f"### 🧠 예측 결과: **{class_name}**")
        st.image(cam_image, caption="Grad-CAM", use_column_width=True)
        st.markdown("---")