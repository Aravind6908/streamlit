import streamlit as st
import shelve
import docx2txt
import PyPDF2
import time  # Used to simulate typing effect
import nltk

from summa.summarizer import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline
nltk.download('punkt')

nltk.download('punkt_tab')

st.set_page_config(page_title="Legal Document Summarizer", layout="wide")

st.title("📄 Legal Document Summarizer (Extractive summary)")

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

# # Extractive Summarization using TextRank
# def extractive_summary(text, num_sentences=3):
#     parser = PlaintextParser.from_string(text, Tokenizer("english"))
#     summarizer = TextRankSummarizer()
#     summary = summarizer(parser.document, num_sentences)
#     return " ".join(str(sentence) for sentence in summary)


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

# Hybrid Summarization: Extractive + Abstractive
def hybrid_summary(text):
    # extractive = extractive_summary(text, num_sentences=5)
    # extractive = lexrank_summary(text, ratio=0.2)
    extractive = lsa_summary(text, num_sentences=5)
    abstractive = abstractive_summary(extractive)
    return f"📌 **Extractive Summary:**\n{extractive}\n\n🔍 **Abstractive Summary:**\n{abstractive}"

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
        file_text = extract_text(uploaded_file)
        # summary_text = hybrid_summary(file_text)
        summary_text = lsa_summary(file_text)
        

        st.session_state.messages.append({
            "role": "user",
            "content": f"📤 Uploaded **{uploaded_file.name}**"
        })

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            preview_text = f"🧾 **Hybrid Summary of {uploaded_file.name}:**\n\n{summary_text}"
            display_with_typing_effect(preview_text, speed=0.005)

        st.session_state.messages.append({
            "role": "assistant",
            "content": preview_text
        })

        st.session_state.last_uploaded = uploaded_file.name
        save_chat_history(st.session_state.messages)
        st.rerun()

# Handle chat input and return hybrid summary
if prompt:
    # summary_text = hybrid_summary(prompt)
    summary_text = lsa_summary(prompt)
    

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
