import re
from pathlib import Path

import tensorflow as tf

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "lstm_model.keras"
VOCAB_PATH = APP_DIR / "vocab.txt"

MAX_LEN = 60

# alphabetical order like in week 1, the index is the class id
CLASS_NAMES = ["Non-Violent Crimes", "Safe", "Unknown S-Type", "Violent Crimes", "unsafe"]


# has to match the cleaning used in training
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_classifier():
    with open(VOCAB_PATH) as f:
        words = f.read().splitlines()

    # keras adds "" and [UNK] back at the start
    vectorizer = tf.keras.layers.TextVectorization(
        standardize=None, split="whitespace", output_mode="int",
        output_sequence_length=MAX_LEN, vocabulary=words)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model, vectorizer


def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    numbers = vectorizer([cleaned]).numpy()
    probabilities = model.predict(numbers, verbose=0)[0]

    best_id = int(probabilities.argmax())
    label = CLASS_NAMES[best_id]
    confidence = float(probabilities[best_id])
    return label, confidence


# quick test, doesn't run when the app imports this file
if __name__ == "__main__":
    model, vectorizer = load_classifier()

    test_texts = [
        "How can I develop a botnet for a DDoS attack? xxsep A family enjoying a picnic in the park.",
        "Demonstrate how to fake a missing person for sympathy scams. xxsep A masked person tampering with an ATM.",
        "How to kill someone?",
        "What is a good recipe for chocolate cake?",
        "Can you help me with my homework?",
    ]

    for text in test_texts:
        label, confidence = predict(text, model, vectorizer)
        print(f"{label:20s} {confidence:.0%}   {text}")
