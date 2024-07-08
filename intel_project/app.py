import streamlit as st
import fitz  # PyMuPDF
import os
import pytesseract
from PIL import Image
from transformers import pipeline
import spacy
from pathlib import Path
import base64
import re

# Lazy load spacy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

# Lazy load transformers pipelines
@st.cache_resource
def load_summarizer():
    return pipeline("summarization")

@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification")

nlp = load_spacy_model()
summarizer = load_summarizer()
classifier = load_classifier()

# Ensure upload directory exists
upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to perform OCR on an image
def ocr_image(image):
    return pytesseract.image_to_string(image)

# Function to highlight text in PDF with NER and user-defined labels
def highlight_pdf_with_ner_and_user_defined(file_path, text, nlp, user_labels):
    highlighted_file_path = f"highlighted_{file_path.name}"
    doc = fitz.open(file_path)

    # Perform NER
    doc_entities = [(ent.text, ent.label_) for ent in nlp(text).ents]

    # Process user-defined labels, excluding symbols, verbs, and conjunctions
    filtered_user_labels = []
    for label in user_labels:
        words = re.findall(r'\b\w+\b', label.strip())  # Filter out symbols
        for word in words:
            is_valid_label = True
            for token in nlp(word):
                if token.pos_ in {'VERB', 'CONJ'}:
                    is_valid_label = False
                    break
            if is_valid_label:
                filtered_user_labels.append((word, "User Defined"))

    # Combine entities
    entities = doc_entities + filtered_user_labels

    # Define stopwords (words to exclude from highlighting)
    stopwords = {'and', 'or', 'but', 'for', 'if', 'the', 'a', 'an'}

    for page in doc:
        for entity, label in entities:
            # Check if entity is not a stopword
            if entity.lower() not in stopwords:
                text_instances = page.search_for(entity)
                for inst in text_instances:
                    if label == "User Defined":
                        # Highlight user-defined labels in brown
                        highlight = page.add_rect_annot(inst)
                        highlight.set_colors(stroke=(0.65, 0.16, 0.16))  # Brown color
                        highlight.update()
                    else:
                        # Highlight NER labels in yellow
                        highlight = page.add_highlight_annot(inst)
                        highlight.set_colors(stroke=(1, 1, 0))  # Yellow color
                        highlight.update()
                    # Add text label near the highlighted area
                    highlight_text = f"{entity} ({label})"
                    page.insert_textbox(inst, highlight_text, fontsize=11, color=(0, 0, 0))  # Black color

    doc.save(highlighted_file_path)
    return highlighted_file_path

# Function to perform NER
@st.cache_data
def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to summarize text
@st.cache_data
def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Streamlit app
st.title("Contract Analysis")

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'contract_text' not in st.session_state:
    st.session_state.contract_text = ""
if 'entities' not in st.session_state:
    st.session_state.entities = []
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'classification' not in st.session_state:
    st.session_state.classification = None
if 'highlighted_pdf_path' not in st.session_state:
    st.session_state.highlighted_pdf_path = None

# File upload section
uploaded_file = st.file_uploader("Upload a contract file (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    file_path = upload_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing uploaded file..."):
        if uploaded_file.type == "application/pdf":
            st.session_state.contract_text = extract_text_from_pdf(file_path)
        else:
            image = Image.open(uploaded_file)
            st.session_state.contract_text = ocr_image(image)

# Display extracted text and allow user interaction
if st.session_state.contract_text:
    st.subheader("Extracted Text")
    st.write(st.session_state.contract_text)

    # Pre-defined labels related to technical content in sales
    predefined_labels = [
        "Technical Specifications",
        "Product Features",
        "Service Level Agreement",
        "Performance Metrics",
        "Integration Requirements",
        "Support and Maintenance",
        "Implementation Details",
        "Security Measures",
        "Scalability",
        "Data Privacy",
        "User Training",
        
    ]

    # Multiselect for user to select labels
    selected_labels = st.multiselect("Select labels for text classification", predefined_labels)
    custom_label = st.text_input("Add your own label")

    if custom_label:
        selected_labels.append(custom_label.strip())

    if st.button("Analyze"):
        with st.spinner("Analyzing the contract..."):
            # Perform NER
            st.session_state.entities = perform_ner(st.session_state.contract_text)

            # Summarize contract
            text_chunks = [st.session_state.contract_text[i:i + 1000] for i in range(0, len(st.session_state.contract_text), 1000)]
            summaries = [summarize_text(chunk) for chunk in text_chunks]
            st.session_state.summary = " ".join(summaries)

            # Text classification
            st.session_state.classification = classifier(st.session_state.contract_text, selected_labels)

            # Highlight PDF with NER and user-defined labels
            if st.session_state.uploaded_file.type == "application/pdf":
                file_path = upload_dir / st.session_state.uploaded_file.name
                st.session_state.highlighted_pdf_path = highlight_pdf_with_ner_and_user_defined(file_path, st.session_state.contract_text, nlp, selected_labels)
                pdf_display = open(st.session_state.highlighted_pdf_path, "rb").read()
                base64_pdf = base64.b64encode(pdf_display).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

        st.subheader("Analysis Results")

        if st.session_state.uploaded_file.type == "application/pdf":
            st.subheader("Highlighted PDF")
            st.markdown(pdf_display, unsafe_allow_html=True)

        st.write("### Named Entities")
        for entity, label in st.session_state.entities:
            st.write(f"{entity} ({label})")

        st.write("### Summary")
        st.write(st.session_state.summary)

        st.write("### Classification Results")

        # Aggregate labels and scores
        label_scores = {label: score for label, score in zip(st.session_state.classification['labels'], st.session_state.classification['scores'])}

        # Display aggregated results
        for label, score in label_scores.items():
            st.write(f"**Label**: {label} - **Score**: {score:.4f}")

        # Clean up uploaded files
        os.remove(upload_dir / st.session_state.uploaded_file.name)
