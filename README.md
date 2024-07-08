# intel-project
Contract Analysis Application....

Overview:

This Streamlit-based application allows users to upload contract files (PDF or images) for analysis. It performs various tasks such as text extraction, entity recognition (NER), summarization, and text classification using pre-defined and user-defined labels.

Features:

File Upload: Upload contract files (PDF or images) for analysis.
Text Extraction: Extract text from uploaded PDFs or perform OCR on images.
Named Entity Recognition (NER): Identify entities such as organizations, dates, and custom-defined labels.
Summarization: Summarize lengthy contract text into concise summaries.
Text Classification: Classify contract content using pre-defined and user-defined labels.
Highlighting: Highlight entities and user-defined labels in uploaded PDFs.

Usage:

Upload: Choose a contract file (PDF or image) using the file uploader.
Analysis: Click the "Analyze" button to perform text extraction, NER, summarization, classification, and PDF highlighting.
Results: View extracted text, named entities, summary, and classification results.

Installation:
To run the application locally:

1.Clone this repository:

git clone https://github.com/your_username/your_repository.git

2.Navigate into the cloned directory:

cd your_repository

3.Install dependencies:

pip install -r requirements.txt

4.Run the Streamlit app:

streamlit run app.py

5.Dependencies:

Streamlit

PyMuPDF (fitz)

pytesseract

Transformers (pipeline)

spacy

6.Contributing:

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and create a pull request with your proposed changes.

