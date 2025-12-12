import os
from dotenv import load_dotenv # type: ignore
import streamlit as st # type: ignore
from datetime import datetime
import time
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain_core.runnables import RunnablePassthrough # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from langchain_groq import ChatGroq # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_chroma import Chroma # type: ignore
import re
import random # Used for dynamic typing simulation

# CONFIGURATION AND ENVIRONMENT SETUP 
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY missing in .env")
    st.stop()

# HARDCODED APP DETAILS 
APP_TITLE = "Communn.io Assistant"
COMPANY_URL = "https://communn.io/"
LOGO_URL = 'https://communn.io/admin/static/media/Logofull.0e1f14991dff53eac56d.png'
VECTOR_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"
RAG_K_DOCS = 8

# UI/THEME CSS 
st.set_page_config(page_title=APP_TITLE, page_icon="🤖")
st.markdown("""
<style>
/* ... (Your CSS from the previous answer remains here) ... */
/* Global styles for a cleaner look */
.stApp {
    background-color: #f0f2f6; /* Light gray background */
}
.stApp header {
    visibility: hidden; /* Hide Streamlit default header */
}
/* Main chat container structure */
.chat-container {
    display: flex;
    flex-direction: column;
    width: 100%;
    max-width: 800px;
    margin: 0 auto;
}
/* User chat bubble - Blue/Green for user, pushed right */
.chat-bubble-user {
    background-color: #007AFF; /* Apple/iOS Blue */
    color: white;
    align-self: flex-end;
    border-radius: 18px 18px 2px 18px;
    margin-left: 20%;
}
/* Bot chat bubble - Light Gray/White, pushed left */
.chat-bubble-bot {
    background-color: #FFFFFF; /* White/Light card color */
    color: #1a1a1a;
    border: 1px solid #e0e0e0;
    align-self: flex-start;
    border-radius: 18px 18px 18px 2px;
    margin-right: 20%;
}
/* Base bubble styling */
.chat-bubble-user, .chat-bubble-bot {
    position: relative;
    padding: 12px 15px 24px 15px; /* Added padding for timestamp */
    margin-bottom: 15px;
    max-width: 80%;
    word-wrap: break-word;
    font-size: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); /* Subtle shadow */
    transition: all 0.3s ease-in-out;
}
.timestamp { 
    font-size: 10px; 
    position: absolute; 
    bottom: 5px; 
    opacity: 0.7; /* Subtler timestamp */
}
.chat-bubble-user .timestamp { 
    right: 10px; 
    color: rgba(255, 255, 255, 0.8);
}
.chat-bubble-bot .timestamp { 
    right: 10px; 
    color: #666;
}
/* Bot typing indicator for dynamic feel */
.typing-indicator {
    display: flex;
    align-items: center;
    background-color: #e0e0e0;
    color: #333;
    padding: 8px 15px;
    border-radius: 18px 18px 18px 2px;
    width: fit-content;
    font-style: italic;
    margin-bottom: 15px;
}
/* Customizing the chat input bar */
div[data-testid="stTextInput"] > div {
    border-radius: 25px; /* Rounded input */
    padding: 5px 15px;
    border: 2px solid #007AFF;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# HELPER FUNCTIONS AND LANGCHAIN PIPELINE

@st.cache_resource(show_spinner=False)   # CRITICAL CHANGE: Cache the RAG components

def setup_rag_components():
    """
    Initializes embeddings, Chroma retriever, and the LLM.
    This function will now run only once per deployed app, 
    significantly reducing latency on subsequent requests.
    """
    try:
        # These operations are slow and are now safely cached.
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": RAG_K_DOCS})
        
        model = ChatGroq(
            model=LLM_MODEL,
            groq_api_key=groq_api_key,
            temperature=0
        )
        return retriever, model
    except Exception as e:
        st.error(f"Error loading RAG components: {e}")
        st.stop()

def format_docs(docs):
    """Formats retrieved documents for the RAG context."""
    return "\n\n".join(d.page_content for d in docs)

# Note: st.cache_data is not used here because the summary depends on session state,
# but using the cached `model` object still makes the function more efficient.
def update_summary(old_summary, new_interaction, llm):
    """Updates the conversation summary for context in the next turn."""
    s_prompt = ChatPromptTemplate.from_template("""
    Summarize the overall conversation history in 2–3 concise sentences, incorporating the new interaction:

    Current Summary:
    {old}

    New Interaction:
    {new}
    """)
    chain = s_prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"old": old_summary, "new": new_interaction})
    except Exception as e:
        # In case of an error, just return the old summary to prevent a crash
        st.warning(f"Error updating summary, maintaining old context: {e}")
        return old_summary

def create_rag_chain(llm):
    """Defines the main RAG chain with the prompt template."""
    prompt = ChatPromptTemplate.from_template("""
    You are a polite, professional, and helpful AI assistant for Communn.io.
    Your main goal is to provide accurate and concise information based on the context and conversation history.
    
    Use the CONTEXT and the SUMMARY to answer the QUESTION accurately.

    ### SUMMARY OF CONVERSATION:
    {summary}

    ### CONTEXT (Retrieved Documents):
    {context}

    ### USER QUESTION:
    {question}
    """)
    
    rag_chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
            "summary": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# SESSION STATE AND INITIALIZATION
if "username" not in st.session_state:
    st.session_state.username = None
if "started" not in st.session_state:
    st.session_state.started = False
if "history" not in st.session_state:
    st.session_state.history = []
if "summary" not in st.session_state:
    st.session_state.summary = "The chat has just started."


# START SCREEN / AUTHENTICATION
def render_welcome_screen():
    # ... (Welcome Screen logic remains the same) ...
    """Renders the professional welcome/login screen."""
    st.markdown(f"""
    <div style='text-align:center; margin-top:50px; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
        <img src='{LOGO_URL}' width='150' style='margin-bottom: 20px;'>
        <h1 style='color: #007AFF;'>{APP_TITLE}</h1> 
        <p style='font-size: 18px; color: #555;'>Your dedicated AI resource for all things at Communn.io.</p>
        <p style='font-size: 16px; margin-top: 30px; color: #555;'> 
            Visit our website: <a href='{COMPANY_URL}' target='_blank' style='color: #007AFF; text-decoration: none;'>{COMPANY_URL}</a>
        </p>
        <div style='margin-top: 30px; text-align: left; max-width: 350px; margin-left: auto; margin-right: auto;'>
            <p>Please enter your name to start the conversation.</p>
    """, unsafe_allow_html=True)
    
    username = st.text_input("Enter your name:", placeholder="Type your name here", label_visibility="collapsed")
    
    if st.button("Start Chatting", use_container_width=True):
        if username and username.strip():
            st.session_state.username = username.strip()
            st.session_state.started = True
            # We must use st.rerun() here to transition from the welcome screen to the chat screen
            st.rerun() 
        else:
            st.error("Please enter a valid name to begin.")
    
    st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.started:
    render_welcome_screen()
    st.stop()


# MAIN CHAT INTERFACE 

# Setup RAG components once the user has started
# This line now calls the CACHED function, which is fast on subsequent runs!
retriever, model = setup_rag_components() 
rag_chain = create_rag_chain(model)

def render_chat_header():
    # ... (Header rendering logic remains the same) ...
    """Renders the main chat header with logo and welcome message, with black text."""
    col1, col2 = st.columns([1, 6])
    
    with col1:
        # Logo container with fixed height for alignment
        st.markdown(f"""
        <div style="height: 80px; display: flex; align-items: center; padding-top: 5px;">
            <img src='{LOGO_URL}' width='80' style='vertical-align: middle; padding-bottom: 3px;'>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Text with dark color
        st.markdown(f"""
            <div style="padding-top: 5px;">
                <h2 style='margin-bottom: 0px; color: #000000;'>Hello {st.session_state.username} 👋</h2>
                <span style='font-size:14px; color:#555555;'>
                    I am your {APP_TITLE}. Ask me anything! | 
                    <a href='{COMPANY_URL}' target='_blank' style='color: #007AFF; text-decoration: none;'>Visit Site</a>
                </span>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

def render_message(is_user, content, timestamp, container=st):
    """Renders a single message bubble into a specified container."""
    css_class = "chat-bubble-user" if is_user else "chat-bubble-bot"
    container.markdown(f"""
        <div class='{css_class}'>
            <div>{content}</div>
            <div class='timestamp'>{timestamp}</div>
        </div>
    """, unsafe_allow_html=True)


# DISPLAY CHAT MESSAGES IN A DYNAMIC CONTAINER 
render_chat_header()

# This is the key change: create a container to append new messages dynamically.
# By default, Streamlit re-renders everything on the page, so we only display
# the current history here. New messages will be written directly below it later.
chat_history_container = st.container()

with chat_history_container:
    # Display all existing history
    for chat in st.session_state.history:
        render_message(
            is_user=True, 
            content=f"**{st.session_state.username}:** {chat['question']}", 
            timestamp=chat['time']
        )
        render_message(
            is_user=False, 
            content=f"**Bot:** {chat['answer']}", 
            timestamp=chat['time']
        )


# CHAT INPUT AND LOGIC (WITHOUT st.rerun())
query = st.chat_input(f"{st.session_state.username}, ask anything about Communn.io...")

if query:
    timestamp = datetime.now().strftime("%H:%M")

    # 1. Immediately render the User's message (optional, but makes it snappier)
    render_message(
        is_user=True, 
        content=f"**{st.session_state.username}:** {query}", 
        timestamp=timestamp
    )
    
    # 2. Show Typing Indicator
    typing_placeholder = st.empty()
    with typing_placeholder:
        st.markdown(
            "<div class='typing-indicator'>🤖 Bot is typing...</div>",
            unsafe_allow_html=True
        )
    
    # Simulate dynamic typing delay 
    time.sleep(random.uniform(0.5, 1.0))

    # 3. RAG/LLM Processing
    # This is fast because `retriever` was loaded from the cache
    docs = retriever.invoke(query) 
    context_text = format_docs(docs)
    
    # Run the RAG chain
    try:
        answer = rag_chain.invoke({
            "context": context_text,
            "question": query,
            "summary": st.session_state.summary,
        })
    except Exception as e:
        answer = f"Apologies, an error occurred while generating the response: {e}. Please try again."

    # Remove the typing indicator
    typing_placeholder.empty()

    # 4. Universal URL Sanitizer
    answer = re.sub(r'https?://[^\s)]+', f'[{COMPANY_URL}]', answer)
    
    # 5. Render the Bot's answer directly (avoids rerun)
    render_message(
        is_user=False, 
        content=f"**Bot:** {answer}", 
        timestamp=timestamp
    )

    # 6. Update History and Summary
    new_entry = f"User asked: {query} | Bot answered: {answer}"
    
    # Update summary in background
    try:
        st.session_state.summary = update_summary(st.session_state.summary, new_entry, model)
    except:
        pass # If summary update fails, we still have the main chat history.

    # Append to history for the next script run
    st.session_state.history.append({
        "question": query,
        "answer": answer,
        "time": timestamp
    })
