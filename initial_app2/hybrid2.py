import streamlit as st
import shelve
import docx2txt
import PyPDF2
import time  # Used to simulate typing effect
import nltk

import re
import os
import requests
from dotenv import load_dotenv


import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import nltk

nltk.download('punkt')


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

st.title("📄 Legal Document Summarizer (Hybrid summary)")

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

# # Clean and normalize text
# def clean_text(text):
#     # Remove extra spaces, line breaks, tabs
#     text = re.sub(r'\s+', ' ', text)
#     text = text.strip()
#     return text

# Clean and normalize text (for legal documents)
def clean_text(text):
    text = text.replace('\r\n', ' ').replace('\n', ' ')  # Replace newlines
    text = re.sub(r'\s+', ' ', text)                    # Replace multiple spaces/tabs
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    text = text.strip()                                 # Trim leading/trailing whitespace
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

# def hf_summary_api(text, model_id, min_length=50, max_length=250):
#     headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
#     payload = {
#         "inputs": text,
#         "parameters": {
#             "min_length": min_length,
#             "max_length": max_length,
#             "do_sample": False
#         }
#     }
#     response = requests.post(
#         f"https://api-inference.huggingface.co/models/{model_id}",
#         headers=headers,
#         json=payload
#     )
#     if response.status_code == 200:
#         return response.json()[0]["summary_text"]
#     else:
#         return f"❌ API Error: {response.status_code} - {response.text}"


# def hybrid_summary_by_section(text):
#     cleaned_text = clean_text(text)
#     sections = section_by_zero_shot(cleaned_text)

#     summary_parts = []
#     for name, content in sections.items():
#         if content.strip():
#             # Extractive using long-form summarizer
#             extractive = hf_summary_api(
#                 content,
#                 model_id="sshleifer/distilbart-cnn-12-6",
#                 max_length=200
#             )

#             # Abstractive using BART
#             abstractive = hf_summary_api(
#                 extractive,
#                 model_id="facebook/bart-large-cnn",
#                 max_length=250
#             )

#             hybrid = f"📌 **Extractive Summary:**\n{extractive}\n\n🔍 **Abstractive Summary:**\n{abstractive}"
#             summary_parts.append(f"### 📘 {name} Section:\n{clean_text(hybrid)}")

#     # return "\n\n".join(summary_parts)
#     return extractive
    # return sections



@st.cache_resource
def load_legalbert():
    return SentenceTransformer("nlpaueb/legal-bert-base-uncased")

@st.cache_resource
def load_bart():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

legalbert_model = load_legalbert()
abstractive_pipeline = load_bart()


def legalbert_extractive_summary(text, top_ratio=0.2):
    sentences = sent_tokenize(text)
    top_k = max(3, int(len(sentences) * top_ratio))

    if len(sentences) <= top_k:
        return text

    # Embeddings & scoring
    sentence_embeddings = legalbert_model.encode(sentences, convert_to_tensor=True)
    doc_embedding = torch.mean(sentence_embeddings, dim=0)
    cosine_scores = util.pytorch_cos_sim(doc_embedding, sentence_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)

    # Preserve original order
    selected_sentences = [sentences[i] for i in sorted(top_results.indices.tolist())]
    return " ".join(selected_sentences)



def bart_abstractive_summary(text, max_length=250, min_length=60):
    summary = abstractive_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']


def hybrid_summary_by_section(text, top_ratio=0.2):
    cleaned_text = clean_text(text)
    sections = section_by_zero_shot(cleaned_text)  # Split into Facts, Arguments, Judgment, Other

    summary_parts = []
    for name, content in sections.items():
        if content.strip():
            # Calculate dynamic number of sentences to extract based on section length
            sentences = sent_tokenize(content)
            top_k = max(3, int(len(sentences) * top_ratio))

            # Extractive summary using Legal-BERT
            extractive = legalbert_extractive_summary(content, 0.2)

            # Abstractive summary using BART
            abstractive = bart_abstractive_summary(extractive)

            # Combine both
            hybrid = f"\ud83d\udccc **Extractive Summary:**\n{extractive}\n\n\ud83d\udd0d **Abstractive Summary:**\n{abstractive}"
            summary_parts.append(f"### \ud83d\udcd8 {name} Section:\n{clean_text(hybrid)}")

    # return "\n\n".join(summary_parts)
    return extractive



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
            display_with_typing_effect(clean_text(preview_text), speed=0.000005)

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
        display_with_typing_effect(clean_text(bot_response), speed=0.000005)

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response
    })

    save_chat_history(st.session_state.messages)
    st.rerun()
