from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_NAME = "Salesforce/blip-image-captioning-base"

# mps is the mac gpu
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# downloads about 1 GB the first time
def load_captioner():
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return processor, model


def generate_caption(image, processor, model):
    # some pngs have a transparency channel, blip needs plain rgb
    image = image.convert("RGB")

    # the input has to be on the same device as the model
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=30)

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption


# caption the test images and classify each caption
if __name__ == "__main__":
    from classifier import load_classifier, predict

    processor, blip_model = load_captioner()
    lstm_model, vectorizer = load_classifier()
    print("BLIP is running on:", DEVICE)

    images_dir = Path(__file__).resolve().parent / "test_images"
    for image_path in sorted(images_dir.glob("*.jpg")):
        image = Image.open(image_path)
        caption = generate_caption(image, processor, blip_model)
        label, confidence = predict(caption, lstm_model, vectorizer)
        print(f"{image_path.name:20s} {caption:50s} -> {label} ({confidence:.0%})")
