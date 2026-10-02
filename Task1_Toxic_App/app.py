import streamlit as st
from PIL import Image

from classifier import load_classifier, predict
from database import create_table, get_all_records, save_record
from imagecaption import generate_caption, load_captioner


# streamlit reruns the whole script on every click, so load the models only once
@st.cache_resource
def get_classifier():
    return load_classifier()


@st.cache_resource
def get_captioner():
    return load_captioner()


st.title("Toxic Content Classifier")

create_table()

lstm_model, vectorizer = get_classifier()
processor, blip_model = get_captioner()

classify_tab, records_tab = st.tabs(["Classify", "All records"])

with classify_tab:
    user_name = st.text_input("Your name")
    user_text = st.text_area("Text to classify (optional)")
    uploaded_image = st.file_uploader("Image (optional)", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, width=300)

    if st.button("Classify"):
        has_text = user_text.strip() != ""
        has_image = uploaded_image is not None

        if user_name.strip() == "":
            st.warning("Please enter your name first.")
        elif not has_text and not has_image:
            st.warning("Please type some text, upload an image, or both.")
        else:
            # stays None for text only
            caption = None
            if has_image:
                image = Image.open(uploaded_image)
                caption = generate_caption(image, processor, blip_model)
                st.info(f"Image caption: {caption}")

            # same format as the training data: query xxsep image description
            if has_text and has_image:
                input_type = "both"
                model_input = user_text + " xxsep " + caption
            elif has_text:
                input_type = "text"
                model_input = user_text
            else:
                input_type = "image"
                model_input = caption

            label, confidence = predict(model_input, lstm_model, vectorizer)
            st.success(f"Result: {label} ({confidence:.0%} confident)")

            saved_text = user_text if has_text else None
            save_record(user_name, input_type, saved_text, caption, label)
            st.caption("Saved to the database.")

# this runs after the save above, so the new row shows up right away
with records_tab:
    records = get_all_records()
    st.write(f"Total records: {len(records)}")
    st.dataframe(records, hide_index=True)

    csv_text = records.to_csv(index=False)
    st.download_button(
        label="Export to CSV",
        data=csv_text,
        file_name="toxic_records.csv",
        mime="text/csv",
    )
