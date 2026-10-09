# Week 2 — Toxic Content App (Streamlit + BLIP) and Quantization Research

Part of my NLP internship at **Cellula Technologies** (Week 2).

## Task 0 — Quantization research
A notebook (`Task0_Quantization/quantization_research.ipynb`) explaining how to make large
models smaller:
- Model compression families: pruning, knowledge distillation, low-rank factorization, quantization.
- Number formats (FP32, FP16, INT8, INT4) and how symmetric / asymmetric quantization works.
- Experiments: quantization error vs number of bits (error roughly doubles for every bit removed),
  and how a single outlier weight makes INT8 error about 8× larger unless per-channel scales are used.
- Post-training dynamic INT8 quantization of a DistilBERT sentiment model: about half the size,
  with almost identical predictions on test sentences.
- Formats used to run LLMs locally, such as GGUF (INT4 / INT8).

## Task 1 — Toxic Content Classifier app
A **Streamlit** web app that reuses my Week 1 LSTM model and adds image support.

**Features**
- Text input, image upload, or both.
- Images are described with **BLIP** (`Salesforce/blip-image-captioning-base`, Hugging Face,
  PyTorch, running on the Mac GPU with `mps`). The caption is then classified by the LSTM.
- Every result is saved in a **SQLite** database (name, input type, text, caption, label, time).
- A second tab shows all records and can export them to CSV.

**Tech stack:** Python, Streamlit, TensorFlow / Keras, PyTorch, Hugging Face Transformers, SQLite, pandas

**Code structure**
```
app.py           Streamlit interface (models cached with st.cache_resource)
classifier.py    loads the LSTM + vocabulary and predicts a label
imagecaption.py  loads BLIP and generates image captions
database.py      SQLite: create table, save record, read all records
```

**How to run**
```bash
cd Task1_Toxic_App
pip install -r requirements.txt
streamlit run app.py
```
