import streamlit as st
import shelve
import docx2txt
import PyPDF2
import time  # Used to simulate typing effect
import nltk

import os
import requests
from dotenv import load_dotenv

from summa.summarizer import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline
from transformers import pipeline
from nltk import sent_tokenize
nltk.download('punkt')

nltk.download('punkt_tab')

st.set_page_config(page_title="Legal Document Summarizer", layout="wide")

st.title("📄 Legal Document Summarizer (Clean and Extractive summary)")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Load chat history
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

# Save chat history
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

# Function to limit text preview to 500 words
def limit_text(text, word_limit=500):
    words = text.split()
    return " ".join(words[:word_limit]) + ("..." if len(words) > word_limit else "")





# CLEAN AND NORMALIZE TEXT

# NEW: Text Preprocessing and Sectioning
import re

# Clean and normalize text
def clean_text(text):
    # Remove extra spaces, line breaks, tabs
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Classification of Document by Sections
# Load zero-shot model only once



#######################################################################################################################


# LOADING MODELS FOR DIVIDING TEXT INTO SECTIONS

# @st.cache_resource
# def load_zero_shot_model():
#     return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
# classifier = load_zero_shot_model()


# Load token from .env file
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")


def classify_zero_shot_hfapi(text, labels):
    if not HF_API_TOKEN:
        return "❌ Hugging Face token not found."

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }

    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": labels
        }
    }

    response = requests.post(
        "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-1",
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        return f"❌ Error from HF API: {response.status_code} - {response.text}"

    result = response.json()
    return result["labels"][0]  # Return the top label


# Labels for section classification
SECTION_LABELS = ["Facts", "Arguments", "Judgment", "Other"]


def classify_chunk(text):
    return classify_zero_shot_hfapi(text, SECTION_LABELS)
    # return result['labels'][0] if result and 'labels' in result else "Other"


# NEW: NLP-based sectioning using zero-shot classification
def section_by_zero_shot(text):
    sections = {"Facts": "", "Arguments": "", "Judgment": "", "Other": ""}
    sentences = sent_tokenize(text)
    chunk = ""

    for i, sent in enumerate(sentences):
        chunk += sent + " "
        if (i + 1) % 3 == 0 or i == len(sentences) - 1:
            label = classify_chunk(chunk.strip())
            print(f"🔎 Chunk: {chunk[:60]}...\n🔖 Predicted Label: {label}")
            # 👇 Normalize label (title case and fallback)
            label = label.capitalize()
            if label not in sections:
                label = "Other"
            sections[label] += chunk + "\n"
            chunk = ""

    return sections



#######################################################################################################################



# EXTRACTING TEXT FROM UPLOADED FILES

# Function to extract text from uploaded file
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file.name.endswith(".docx"):
        full_text = docx2txt.process(file)
    elif file.name.endswith(".txt"):
        full_text = file.read().decode("utf-8")
    else:
        return "Unsupported file type."
    
    return full_text  # Full text is needed for summarization


#######################################################################################################################



# EXTRACTIVE AND ABSTRACTIVE SUMMARIZATION

 # Extractive Summarization using LSA
def lsa_summary(text, num_sentences=20):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# Extractive Summarization using LexRank
def lexrank_summary(text, ratio=0.2):
    """
    Perform LexRank-based extractive summarization.

    :param text: Full input text.
    :param ratio: Fraction of sentences to keep (e.g., 0.2 = 20%).
    :return: Extractive summary.
    """
    try:
        summary = summarize(text, ratio=ratio)
        return summary.strip()
    except ValueError:
        return "⚠️ Text too short to summarize with LexRank."

# Abstractive Summarization using Hugging Face Transformers
def abstractive_summary(text):
    summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarization_pipeline(text, max_length=200, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# # Hybrid Summarization: Extractive + Abstractive
# def hybrid_summary(text):
#     # extractive = extractive_summary(text, num_sentences=5)
#     # extractive = lexrank_summary(text, ratio=0.2)
#     extractive = lsa_summary(text, num_sentences=5)
#     abstractive = abstractive_summary(extractive)
#     return f"📌 **Extractive Summary:**\n{extractive}\n\n🔍 **Abstractive Summary:**\n{abstractive}"


# Updated hybrid summary section-wise (LSA-based for now)
def hybrid_summary_by_section(text):
    cleaned_text = clean_text(text)
    sections = section_by_zero_shot(cleaned_text)  # 👈 uses zero-shot classifier

    summary_parts = []
    for name, content in sections.items():
        if content.strip():
            extractive = lsa_summary(content, num_sentences=4)
            summary_parts.append(f"### 📘 {name} Section:\n{extractive}")

    # return "\n\n".join(summary_parts)
    return sections


#######################################################################################################################


# STREAMLIT APP INTERFACE CODE

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Initialize last_uploaded if not set
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# Sidebar with a button to delete chat history
with st.sidebar:
    st.subheader("⚙️ Options")
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        st.session_state.last_uploaded = None
        save_chat_history([])

# Display chat messages with a typing effect
def display_with_typing_effect(text, speed=0.005):
    placeholder = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text)
        time.sleep(speed)
    return displayed_text

# Show existing chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Standard chat input field
prompt = st.chat_input("Type a message...")

# Standard File Upload (Below Chat Input)
uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# Handle file upload and generate hybrid summary
if uploaded_file:
    if uploaded_file.name != st.session_state.last_uploaded:
        # file_text = extract_text(uploaded_file)
        raw_text = extract_text(uploaded_file)
        summary_text = hybrid_summary_by_section(raw_text)

        st.session_state.messages.append({
            "role": "user",
            "content": f"📤 Uploaded **{uploaded_file.name}**"
        })

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            preview_text = f"🧾 **Hybrid Summary of {uploaded_file.name}:**\n\n{summary_text}"
            display_with_typing_effect(preview_text, speed=0.000005)

        st.session_state.messages.append({
            "role": "assistant",
            "content": preview_text
        })

        st.session_state.last_uploaded = uploaded_file.name
        save_chat_history(st.session_state.messages)
        st.rerun()

# Handle chat input and return hybrid summary
if prompt:
    raw_text = prompt
    summary_text = hybrid_summary_by_section(raw_text)
    
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        bot_response = f"📝 **Hybrid Summary of your text:**\n\n{summary_text}"
        display_with_typing_effect(bot_response, speed=0.00009)

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response
    })

    save_chat_history(st.session_state.messages)
    st.rerun()
