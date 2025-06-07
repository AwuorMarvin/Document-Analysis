import streamlit as st
import PyPDF2
import docx
import re
from collections import Counter
import pandas as pd
import plotly.express as px
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from textblob import TextBlob
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        st.error(f"Error reading Word document: {e}")
        return ""


def process_file(uploaded_file):
    if uploaded_file.name.lower().endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.lower().endswith(('.docx', '.doc')):
        return extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format.")
        return ""


def letter_frequency_analysis(text):
    lower = Counter(c for c in text if c.islower())
    upper = Counter(c for c in text if c.isupper())
    letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    return pd.DataFrame({
        "Letter": letters,
        "Lowercase": [lower.get(l, 0) for l in letters],
        "Uppercase": [upper.get(l.upper(), 0) for l in letters]
    })


def paragraph_word_count_analysis(text):
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    counts = [len(re.findall(r'\b\w+\b', para)) for para in paragraphs]
    return paragraphs, counts


def number_distribution_analysis(text):
    digits = re.findall(r'\b[0-9]\b', text)
    counter = Counter(digits)
    return pd.DataFrame.from_dict({str(i): counter.get(str(i), 0) for i in range(10)},
                                   orient='index', columns=['Count']).sort_index()


def special_character_analysis(text):
    specials = [c for c in text if not c.isalnum() and not c.isspace() and c.isprintable()]
    return pd.DataFrame.from_dict(Counter(specials), orient='index', columns=['Count']).sort_index()


@st.cache_resource(show_spinner="Loading QA model and indexing document...")
def get_local_qa_chain(text):
    try:
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([text])
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(docs, embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        model_name = "google/flan-t5-base"  # Lightweight and works well for QA
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
        llm = HuggingFacePipeline(pipeline=pipe)

        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    except Exception as e:
        st.error("Failed to load QA model or vector DB.")
        logger.exception("Error initializing QA chain")
        return None

def main():
    st.set_page_config(page_title="Document Analyzer", layout="wide")
    st.title("📄 Document Analysis Tool")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    uploaded_file = st.file_uploader("Upload a PDF or Word file", type=["pdf", "docx", "doc"])

    if uploaded_file:
        text = process_file(uploaded_file)
        if not text:
            return

        st.subheader("Text Preview")
        st.text_area("Extracted Text", text[:500] + "..." if len(text) > 500 else text, height=200)

        # Document Summary
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        total_words = len(words)
        paragraphs, word_counts = paragraph_word_count_analysis(text)
        reading_time = max(1, int(round(total_words / 200)))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Words", f"{total_words:,}")
        c2.metric("Paragraphs", f"{len(paragraphs):,}")
        c3.metric("Characters", f"{len(text):,}")
        c4.metric("Reading Time", f"{reading_time} min")

        # Paragraph Analysis
        st.subheader("Paragraph Word Count & Sentiment")
        col1, col2 = st.columns(2)

        with col1:
            df = pd.DataFrame({'Paragraph': [f"Para {i+1}" for i in range(len(paragraphs))], 'Word Count': word_counts})
            st.dataframe(df)
            hist = pd.DataFrame({'Word Count': word_counts})
            chart = alt.Chart(hist).mark_bar().encode(
                x=alt.X('Word Count:Q', bin=alt.Bin(maxbins=20)),
                y='count()'
            ).properties(title="Paragraph Length Distribution", width=400)
            st.altair_chart(chart)

        with col2:
            sentiment_labels = []
            sentiment_data = []
            for i, para in enumerate(paragraphs):
                polarity = TextBlob(para).sentiment.polarity
                if polarity > 0.1:
                    sentiment = "Positive"
                elif polarity < -0.1:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
                sentiment_labels.append(sentiment)
                sentiment_data.append({"Paragraph": f"Para {i+1}", "Polarity Score": polarity, "Sentiment": sentiment})
            st.dataframe(pd.DataFrame(sentiment_data))

        # Character Analysis
        st.subheader("Character & Digit Analysis")
        col3, col4 = st.columns(2)

        with col3:
            df_letters = letter_frequency_analysis(text)
            melted = pd.melt(df_letters, id_vars='Letter', value_vars=['Lowercase', 'Uppercase'],
                             var_name='Case', value_name='Count')
            chart = alt.Chart(melted).mark_bar().encode(
                x='Letter:N',
                y='Count:Q',
                color='Case:N'
            ).properties(title="Letter Frequency", width=400)
            st.altair_chart(chart)

        with col4:
            df_special = special_character_analysis(text).reset_index().rename(columns={'index': 'Character'})
            if not df_special.empty:
                fig = px.bar(df_special, x='Character', y='Count', title="Special Character Distribution")
                st.plotly_chart(fig)

        # Digits and Sentiment
        st.subheader("Digit & Sentiment Distribution")
        col5, col6 = st.columns(2)

        with col5:
            df_digits = number_distribution_analysis(text).reset_index().rename(columns={'index': 'Digit'})
            fig = px.bar(df_digits, x='Digit', y='Count', title="Digit Distribution")
            st.plotly_chart(fig)

        with col6:
            sentiment_counts = pd.Series(sentiment_labels).value_counts()
            fig = px.pie(
                names=sentiment_counts.index,
                values=sentiment_counts.values,
                title="Sentiment Distribution",
                hole=0.4
            )
            st.plotly_chart(fig)

                # Chat Interface in Sidebar
        with st.sidebar:
            st.header("🧠 Chat with the Document")

            user_question = st.text_input("Ask a question")

            if user_question:
                qa_chain = get_local_qa_chain(text)
                if qa_chain is None:
                    st.error("QA system failed to initialize.")
                else:
                    with st.spinner("Thinking..."):
                        try:
                            response = qa_chain.run(user_question)
                            st.success("Answer generated!")
                            st.session_state.chat_history.append((user_question, response))
                        except Exception as e:
                            st.error("Error during QA response.")
                            logger.exception("Error generating QA response")

            if st.session_state.chat_history:
                st.markdown("### 💬 Chat History")
                for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
                    st.markdown(f"**Q{i+1}:** {q}")
                    st.markdown(f"**A{i+1}:** {a}")

if __name__ == "__main__":
    main()
