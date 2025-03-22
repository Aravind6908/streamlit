# from openai import OpenAI
import streamlit as st
# from dotenv import load_dotenv
import shelve
import docx2txt
import PyPDF2
import time  # Used to simulate typing effect


# load_dotenv()

st.set_page_config(page_title="Legal Document Summarizer", layout="wide")

st.title("📄 Legal Document Summarizer")


USER_AVATAR = "👤"
BOT_AVATAR = "🤖"
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Ensure openai_model is initialized in session state
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"


# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])


# Save chat history to shelve file
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
    
    return limit_text(full_text)  # Limit text to 500 words
    

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
def display_with_typing_effect(text):
    placeholder = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text)
        time.sleep(0.00005)  # Simulate slow typing effect like ChatGPT
    return displayed_text

    
# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Standard chat input field (No send button)
prompt = st.chat_input("Type a message...")

# Standard File Upload (Outside Chat Input)
uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])


# Handle file upload and extract text
if uploaded_file:
    if uploaded_file.name != st.session_state.last_uploaded:
        file_text = extract_text(uploaded_file)

        st.session_state.messages.append({
            "role": "user",
            "content": f"📤 Uploaded **{uploaded_file.name}**"
        })

        # Simulate bot typing effect
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            preview_text = f"🧾 **Preview of {uploaded_file.name} (first 500 words):**\n\n{file_text}"
            display_with_typing_effect(preview_text)

        # Store the message
        st.session_state.messages.append({
            "role": "assistant",
            "content": preview_text
        })

        st.session_state.last_uploaded = uploaded_file.name
        save_chat_history(st.session_state.messages)
        st.rerun()


# Handle chat input and return preview (500 words) with typing effect
if prompt:
    preview_text = limit_text(prompt)

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Simulate bot typing effect
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        bot_response = f"📝 **Preview of your document text (first 500 words):**\n\n{preview_text}"
        display_with_typing_effect(bot_response)

    # Store the message
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response
    })

    save_chat_history(st.session_state.messages)
    st.rerun()