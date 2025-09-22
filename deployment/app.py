import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.model import ScratchCNN  # import your CNN class

# -----------------------------
# Add models folder to path (one level up)
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# -----------------------------
# Load model
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH      = os.path.join(MODELS_DIR, "best_model.pth")

num_classes = 7
model = ScratchCNN(num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# Define transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -----------------------------
# Emotion labels
# -----------------------------
fer_emotions = ["angry","disgust","fear","happy","neutral","sad","surprise"]

# -----------------------------
# Prediction function
# -----------------------------
def predict_emotion(image):
    img = transform(image).unsqueeze(0).to(DEVICE)  # batch size 1
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        top_idx = torch.argmax(probs, dim=1).item()
        emotion = fer_emotions[top_idx]
        probability = probs[0, top_idx].item()
    return emotion, probability

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Emotion Detector")
st.write("Upload a face image and detect emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Emotion"):
        emotion, prob = predict_emotion(image)
        st.success(f"Detected Emotion: **{emotion}** ({prob*100:.2f}%)")
