import spaces
import torch
import torch.nn as nn
import gradio as gr
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

# class_names = sorted(df["dx"].unique()) from the notebook
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Human-readable labels for the UI (order matches CLASS_NAMES)
CLASS_LABELS = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma (akiec)",
    "bcc": "Basal cell carcinoma (bcc)",
    "bkl": "Benign keratosis-like lesions (bkl)",
    "df": "Dermatofibroma (df)",
    "mel": "Melanoma (mel)",
    "nv": "Melanocytic nevi (nv)",
    "vasc": "Vascular lesions (vasc)",
}

MODEL_PATH = "efficientnet_b2_ham10000.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

weights0 = EfficientNet_B2_Weights.DEFAULT
model = efficientnet_b2(weights=None)  # architecture only, we load our own weights next
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features=1408, out_features=len(CLASS_NAMES)),
)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
preprocess = weights0.transforms()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@spaces.GPU(duration=30)
@torch.inference_mode()
def predict(image):
    if image is None:
        return None

    image = image.convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    return {CLASS_LABELS[CLASS_NAMES[i]]: float(probs[i]) for i in range(len(CLASS_NAMES))}


DISCLAIMER = """
**⚠️ Not a medical device.** This model was trained and evaluated on the HAM10000
dataset only, has not been tested on external data, and has not been clinically
validated. It is an educational / portfolio project and must not be used to
diagnose, rule out, or make decisions about any real skin condition. If you
have a concerning skin lesion, see a dermatologist.
"""

with gr.Blocks(title="HAM10000 Skin Lesion Classifier") as demo:
    gr.Markdown("# HAM10000 Skin Lesion Classifier (EfficientNet-B2)")
    gr.Markdown(
        "Upload a dermatoscopic image of a skin lesion to get predicted class "
        "probabilities across the 7 HAM10000 diagnostic categories."
    )
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Lesion image")
            submit_btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            output_labels = gr.Label(num_top_classes=7, label="Predicted probabilities")

    submit_btn.click(fn=predict, inputs=image_input, outputs=output_labels)
    image_input.change(fn=predict, inputs=image_input, outputs=output_labels)

if __name__ == "__main__":
    demo.launch()