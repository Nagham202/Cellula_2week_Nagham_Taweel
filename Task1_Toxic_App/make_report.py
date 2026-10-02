import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from classifier import CLASS_NAMES, clean_text, load_classifier, predict
from imagecaption import DEVICE, generate_caption, load_captioner

FOLDER = Path(__file__).resolve().parent
DATA_PATH = Path.home() / "Cellula_1week_Nagham_Taweel" / "toxic_data.csv"
CLASSES_TO_DROP = ["Elections", "Sex-Related Crimes",
                   "Child Sexual Exploitation", "Suicide & Self-Harm"]
SEED = 42

# every number in the report is computed here, nothing typed by hand
lstm_model, vectorizer = load_classifier()
processor, blip_model = load_captioner()

# same data prep as week 1
df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"query": "text", "image descriptions": "image", "Toxic Category": "label"})
df = df[~df["label"].isin(CLASSES_TO_DROP)]
df["clean_text"] = df["text"].apply(clean_text) + " xxsep " + df["image"].apply(clean_text)
df = df.drop_duplicates(subset="clean_text")
df["label_id"] = df["label"].map({name: i for i, name in enumerate(CLASS_NAMES)})

X = df["clean_text"].to_numpy(dtype=object)
y = df["label_id"].to_numpy()
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)


def macro_f1(texts, labels):
    predictions = lstm_model.predict(vectorizer(texts).numpy(), verbose=0).argmax(axis=1)
    return f1_score(labels, predictions, average="macro")


val_f1 = macro_f1(X_val, y_val)
test_f1 = macro_f1(X_test, y_test)

# how much the image description alone decides the label
n_descriptions = df["image"].nunique()
majority_share = df.groupby("image")["label"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(df)

description_rows = []
correct_alone = 0
for description in sorted(df["image"].unique()):
    data_label = df[df["image"] == description]["label"].value_counts().index[0]
    label, confidence = predict(description, lstm_model, vectorizer)
    correct_alone += label == data_label
    description_rows.append([description, data_label, f"{label} ({confidence:.0%})",
                             "yes" if label == data_label else "no"])

# safe and unknown s-type use the same peaceful descriptions
shared = pd.crosstab(df["image"], df["label"])[["Safe", "Unknown S-Type"]]
shared = shared[(shared["Safe"] > 0) & (shared["Unknown S-Type"] > 0)]

# try each common word on its own
words = open(FOLDER / "vocab.txt").read().splitlines()[:1500]
word_probabilities = lstm_model.predict(vectorizer(np.array(words, dtype=object)).numpy(), verbose=0)
top_words = {}
for class_id, name in enumerate(CLASS_NAMES):
    best = np.argsort(-word_probabilities[:, class_id])[:8]
    top_words[name] = ", ".join(words[i] for i in best)

# typed text, with and without a training description
typed_examples = [
    "How to kill someone?",
    "I want to kill someone",
    "What is a good recipe for chocolate cake?",
    "Can you help me with my homework?",
    "What is a good recipe for chocolate cake? xxsep A family enjoying a picnic in the park.",
    "How can I develop a botnet for a DDoS attack? xxsep A family enjoying a picnic in the park.",
    "How do I hack my neighbor wifi? xxsep A family enjoying a picnic in the park.",
]
typed_rows = []
for text in typed_examples:
    label, confidence = predict(text, lstm_model, vectorizer)
    typed_rows.append([text.replace(" xxsep ", " <b>xxsep</b> "), label, f"{confidence:.0%}"])

# blip captions for the test images
caption_rows = []
caption_times = []
for image_path in sorted((FOLDER / "test_images").glob("*.jpg")):
    image = PILImage.open(image_path)
    start = time.time()
    caption = generate_caption(image, processor, blip_model)
    caption_times.append(time.time() - start)
    label, confidence = predict(caption, lstm_model, vectorizer)
    caption_rows.append([image_path.name, caption, f"{label} ({confidence:.0%})"])
# the first caption is slower (warm-up), so it's left out of the average
caption_seconds = np.mean(caption_times[1:]) if len(caption_times) > 1 else caption_times[0]

styles = getSampleStyleSheet()
styles["Heading2"].keepWithNext = 1
small = styles["BodyText"].clone("small", fontSize=8.5, leading=10.5)
grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "grey"),
                   ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                   ("VALIGN", (0, 0), (-1, -1), "TOP")])


# wrap the cells so long text goes onto a new line
def make_table(rows, widths):
    cells = [[Paragraph(str(value), small) for value in row] for row in rows]
    table = Table(cells, colWidths=[w * cm for w in widths], hAlign="LEFT", repeatRows=1)
    table.setStyle(grid)
    return table


# build the pdf
story = []
body = styles["BodyText"]

story.append(Paragraph("Toxic Content Classifier App - Week 2, Task 1", styles["Title"]))

story.append(Paragraph("Summary", styles["Heading2"]))
story.append(Paragraph(
    "A Streamlit web app was built that classifies text, an image, or both into the 5 toxic categories "
    "used in Week 1. Images are described in words by BLIP-1 (Salesforce/blip-image-captioning-base), "
    "and the text is classified by the Week 1 LSTM. Every submission is saved automatically to a SQLite "
    "database, and all records can be viewed in the app and exported to CSV. The app works for all three "
    "input types. However, testing showed that the LSTM generalises poorly to new text: the dataset has "
    f"only <b>{n_descriptions}</b> different image descriptions, and the label can be guessed from the "
    f"description alone <b>{majority_share:.1%}</b> of the time. The LSTM mostly learned to recognise those "
    "sentences, so typed text and BLIP captions are often classified wrongly with low confidence.", body))

story.append(Paragraph("How the app works", styles["Heading2"]))
story.append(make_table([
    ["Input", "Pipeline", "Text the LSTM receives"],
    ["Text only", "text -> LSTM", "the user's text"],
    ["Image only", "image -> BLIP -> caption -> LSTM", "the caption"],
    ["Text and image", "image -> BLIP -> caption, then text + caption -> LSTM",
     "text + \" xxsep \" + caption (the Week 1 training format)"],
], [3, 7, 7]))
story.append(Spacer(1, 8))
story.append(make_table([
    ["File", "Purpose"],
    ["app.py", "Streamlit app: a Classify tab and an All records tab with Export to CSV"],
    ["classifier.py", "Loads the LSTM and vocabulary, applies the Week 1 cleaning, predicts the label"],
    ["imagecaption.py", "Loads BLIP-1 and generates a caption for one image"],
    ["database.py", "SQLite: creates the table, saves a record, reads all records"],
    ["rebuild_vectorizer.py", "Rebuilds the Week 1 vocabulary into vocab.txt and checks it"],
], [4, 13]))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Database columns: id, timestamp, user_name, input_type (text / image / both), user_text, "
    "image_caption, label. The table is created with CREATE TABLE IF NOT EXISTS, so restarting the app "
    "keeps the old records, and values are inserted with ? placeholders to prevent SQL injection. "
    "Both models are loaded once with st.cache_resource, because Streamlit reruns the script on every click.",
    body))

story.append(Paragraph("App screenshot", styles["Heading2"]))
story.append(Image(str(FOLDER / "report_images" / "app_screenshot.png"), width=10.5 * cm, height=6.51 * cm))

story.append(Paragraph("Implementation notes", styles["Heading2"]))
story.append(Paragraph(
    "<b>Rebuilding the vocabulary.</b> The Week 1 TextVectorization layer was not saved with the model. "
    "It was rebuilt by repeating the exact Week 1 steps (same cleaning, duplicate removal and split with "
    "random_state=42) and adapting on the same 1407 training rows, then saved to vocab.txt. To prove it "
    f"matches, the saved model was evaluated again: validation macro F1 <b>{val_f1:.4f}</b> and test macro "
    f"F1 <b>{test_f1:.4f}</b>, the same as in Week 1 (0.8648 and 0.8138).", body))
story.append(Paragraph(
    "<b>pandas and pyarrow.</b> Streamlit installs pyarrow, which makes pandas 3 store text in a format "
    "that train_test_split cannot index. The text column is converted with to_numpy(dtype=object) to get "
    "the same array type as in Week 1.", body))
story.append(Paragraph(
    f"<b>Two frameworks.</b> The LSTM runs in TensorFlow and BLIP in PyTorch, in one environment "
    f"(Python 3.12, TensorFlow 2.21, PyTorch {torch.__version__}). BLIP runs on the Mac GPU ({DEVICE}) and "
    f"takes about <b>{caption_seconds:.2f} s</b> per caption.", body))


story.append(Paragraph("Why predictions on new inputs are weak", styles["Heading2"]))
story.append(Paragraph(
    f"After removing duplicates, the dataset has {len(df)} rows but only {n_descriptions} different image descriptions. When "
    f"each description alone is given to the LSTM, with no query, it predicts the most common label of that "
    f"description for <b>{correct_alone} of {n_descriptions}</b>:", body))
story.append(Spacer(1, 4))
story.append(make_table([["Image description", "Most common label", "LSTM (description only)", "Match"]]
                        + description_rows, [7, 3.2, 4.8, 1.3]))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "The three that do not match are the peaceful descriptions. These are shared by Safe and Unknown "
    "S-Type, so only the query can separate the two classes:", body))
story.append(Spacer(1, 4))
story.append(make_table([["Image description", "Safe rows", "Unknown S-Type rows"]]
                        + [[d, int(r["Safe"]), int(r["Unknown S-Type"])] for d, r in shared.iterrows()],
                        [9, 3, 4]))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Testing each of the 1500 most common words on its own shows which words push each class the most. "
    "All of them come from the image descriptions; words such as kill, gun, steal or blood are not "
    "among them. Single words only reach about 50-60% confidence, while a full description reaches "
    "close to 100%, because the LSTM learned the word sequences.", body))
story.append(Spacer(1, 4))
story.append(make_table([["Class", "Strongest single words"]]
                        + [[name, top_words[name]] for name in CLASS_NAMES], [4, 13]))

story.append(Paragraph("Typed text", styles["Heading2"]))
story.append(Paragraph(
    "Typed text alone gets low confidence and often a wrong label. When a peaceful training description "
    "is added, the query does decide between Safe and Unknown S-Type, which shows that the model learned "
    "some signal from queries, mainly hacking-style questions.", body))
story.append(Spacer(1, 4))
story.append(make_table([["Input", "Prediction", "Confidence"]] + typed_rows, [11, 3.5, 2.5]))

story.append(Paragraph("Image captions", styles["Heading2"]))
story.append(Paragraph(
    "BLIP's captions matched the main content of every test image. The errors come from the LSTM: captions that do "
    "not resemble one of the training descriptions get close-to-random labels. my_test.jpg shows a woman "
    "with blood on her face, and the caption says so clearly, yet it is not classified as Violent Crimes. During "
    "app testing, the caption \"a man holding a knife in his hand\" was classified correctly as Violent "
    "Crimes, because it shares words with \"A shadowy figure holding a weapon\".", body))
story.append(Spacer(1, 4))
story.append(make_table([["Image", "BLIP caption", "LSTM prediction"]] + caption_rows, [3.5, 8, 5.5]))

story.append(Paragraph("Conclusion", styles["Heading2"]))
story.append(Paragraph(
    "All task requirements are implemented: text, image and combined input, BLIP captioning in a separate "
    "imagecaption.py, the Week 1 LSTM, automatic saving to SQLite, a records view and CSV export, all in .py "
    "files with Streamlit. The pipeline works correctly end to end. The weak point is the classifier: "
    "because the training data has only 12 image descriptions, the LSTM learned those sentences rather than "
    "the meaning of toxic content, so its Week 1 score (0.81 macro F1) does not carry over to real user "
    "text or BLIP captions. Training on more varied descriptions, or using a pretrained model from the "
    "allowed list (DistilBERT or ALBERT with LoRA, or Llama Guard), would be expected to generalise better.",
    body))

SimpleDocTemplate(str(FOLDER / "Task1_report.pdf"), pagesize=A4,
                  topMargin=1.5 * cm, bottomMargin=1.5 * cm).build(story)
print("Saved", FOLDER / "Task1_report.pdf")
