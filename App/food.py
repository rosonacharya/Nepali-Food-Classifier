import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# food classes
class_names = [
    'Burger', 'Chiya', 'Dalbhat', 'Friedrice', 'Jeri',
    'Momo', 'Omelette', 'Pakoda', 'Panipuri', 'Pizza',
    'Roti', 'Samosa', 'Selroti', 'Yomari', 'chatamari',
    'chhoila', 'dhindo', 'gundruk', 'kheer', 'sekuwa'
]

num_classes = len(class_names)

# page config
st.set_page_config(
    page_title="Nepali Food Classifier",
    page_icon="🍲",
    layout="centered"
)

# load model
@st.cache_resource
def load_model():

    model = models.convnext_tiny(weights=None)

    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        num_classes
    )

    model.load_state_dict(
        torch.load("convnext.pth", map_location=device)
    )

    model.to(device)
    model.eval()

    return model

# initialize model
model = load_model()

# image transforms
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# title
st.title("🍲 Nepali Food Classifier 🍲 ")

st.write(
    "Upload a food image or take a picture using your camera."
)

# choose input source
option = st.radio(
    "Choose Image Source",
    ["Upload Image", "Use Camera"]
)

image = None

# upload image option
if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Food Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

# camera option
elif option == "Use Camera":

    camera_photo = st.camera_input(
        "Take a Food Picture"
    )

    if camera_photo is not None:

        image = Image.open(camera_photo).convert("RGB")

# prediction
if image is not None:

    # display image
    st.image(
        image,
        caption="Selected Image",
        width=400
    )

    # preprocess image
    img = test_tf(image).unsqueeze(0).to(device)

    # prediction
    with st.spinner("Predicting..."):

        with torch.no_grad():

            outputs = model(img)

            probs = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probs, 1)

    # predicted class
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100

    # show prediction
    st.success(
        f"Predicted Food: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence_score:.2f}%"
    )