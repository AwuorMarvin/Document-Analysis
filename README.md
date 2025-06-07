# Document Analysis App

This Streamlit app allows you to upload a PDF or Word document and provides a comprehensive analysis of its content. Features include:

- **Text Extraction:** Supports both PDF and DOCX files.
- **Document Metrics:** Displays total words, paragraphs, characters, and estimated reading time.
- **Paragraph Analysis:** Shows word count and sentiment (positive, negative, neutral) for each paragraph.
- **Character Analysis:** Visualizes letter frequency (upper/lowercase) and special character distribution.
- **Digit Analysis:** Shows the distribution of single digits (0-9) in the document.
- **Word Cloud & Top Words:** Displays a word cloud and table of the most frequent words.
- **Interactive Chat:** Ask questions about your document using a local language model (no external API required). The chat interface is in the sidebar, with visually distinct chat bubbles for user and AI responses.

## How to Use

1. Upload a PDF or Word document.
2. View the automatic analysis and visualizations.
3. Use the sidebar chat to ask questions about your document.

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

## Getting Started

```sh
pip install -r requirements.txt
streamlit run src/app.py
```

## Notes

- The app runs all analysis locally; no data is sent to external servers.
- Make sure your `.venv` or other virtual environment folders are excluded from git using `.gitignore`.

---