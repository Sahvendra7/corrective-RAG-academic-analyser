import streamlit as st
import requests
import json
import uuid

st.set_page_config(page_title="CRAG Agent", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
<style>
    /* Premium Dark Theme Variables */
    :root {
        --bg-color: #0f111a;
        --secondary-bg: #1e2130;
        --accent-color: #6366f1;
        --accent-gradient: linear-gradient(135deg, #6366f1 0%, #d946ef 100%);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main App Background */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-bg) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    /* Chat Messages Container */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stChatMessage"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -2px rgba(0, 0, 0, 0.3);
    }

    /* User Message Specific */
    [data-testid="stChatMessage"][data-testid="chat-message-user"] {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.15);
    }

    /* Assistant Message Specific */
    [data-testid="stChatMessage"][data-testid="chat-message-assistant"] {
        background: rgba(217, 70, 239, 0.05);
        border: 1px solid rgba(217, 70, 239, 0.15);
    }

    /* Input Box */
    [data-testid="stChatInput"] {
        background: var(--secondary-bg);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text-primary) !important;
    }
    
    /* Thinking Animation */
    @keyframes pulse-glow {
        0% { opacity: 0.6; text-shadow: 0 0 5px rgba(99, 102, 241, 0.2); }
        50% { opacity: 1; text-shadow: 0 0 15px rgba(99, 102, 241, 0.8); }
        100% { opacity: 0.6; text-shadow: 0 0 5px rgba(99, 102, 241, 0.2); }
    }
    .thinking-text {
        animation: pulse-glow 1.5s infinite ease-in-out;
        color: var(--accent-color);
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.95rem;
        padding: 8px 12px;
        background: rgba(99, 102, 241, 0.1);
        border-radius: 8px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        width: fit-content;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("Welcome to the **Premium CRAG Agent**.")
    st.markdown("This agent uses a Corrective Retrieval-Augmented Generation pipeline to ensure accuracy and reduce hallucinations.")
    
    st.divider()
    
    st.markdown("### Agent Status")
    
    import os
    api_url = os.getenv("CRAG_API_URL", "http://localhost:8000")
    
    try:
        # Quick ping to see if server is up
        requests.get(api_url, timeout=0.5)
        is_online = True
    except:
        is_online = False
        
    if is_online:
        st.markdown("<div style='display: flex; align-items: center; gap: 8px;'><span style='height: 10px; width: 10px; background-color: #10b981; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #10b981;'></span> <span style='font-weight: 600;'>System Online</span></div>", unsafe_allow_html=True)
        st.caption("Connected to LangGraph Backend")
    else:
        st.markdown("<div style='display: flex; align-items: center; gap: 8px;'><span style='height: 10px; width: 10px; background-color: #ef4444; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #ef4444;'></span> <span style='font-weight: 600; color: #ef4444;'>Agent Offline</span></div>", unsafe_allow_html=True)
        st.caption("Cannot connect to FastAPI server")
    
    st.divider()
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("✨ Corrective RAG Agent")
st.markdown("Ask a question and watch the LangGraph agent self-correct and reason in real-time. Experience the premium UI.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Generate a unique session ID per user session (fixes cross-user contamination)
if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit-{uuid.uuid4().hex[:12]}"

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a machine learning research question..."):
    
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        status_text = st.empty()
        answer_placeholder = st.empty()
        final_answer = ""
        
        # Connect to your FastAPI server
        import os
        api_url = os.getenv("CRAG_API_URL", "http://localhost:8000")
        url = f"{api_url}/chat/stream"
        payload = {
            "query": prompt, 
            "thread_id": st.session_state.session_id
        }
        
        try:
            # stream=True is crucial for reading the Server-Sent Events
            # timeout=(connect_timeout, read_timeout) prevents indefinite hangs
            with requests.post(url, json=payload, stream=True, timeout=(5, 120)) as response:
                response.raise_for_status()
                
                # Iterate over the streaming chunks from FastAPI
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        
                        # Parse SSE format "data: {...}"
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[len("data: "):]
                            
                            try:
                                data = json.loads(data_str)
                                event = data.get("event")
                                
                                if event == "node_update":
                                    node = data.get("node")
                                    updates = data.get("state_updates", {})
                                    
                                    # Show the user what the agent is thinking
                                    status_html = f"<div class='thinking-text'>⚡ Processing... <code>[ {node} ]</code> node active</div>"
                                    status_text.markdown(status_html, unsafe_allow_html=True)
                                    
                                    # If we hit the generation or hallucination node, display the text
                                    if node in ["generator", "hallucination_checker"]:
                                        if "generation" in updates:
                                            final_answer = updates["generation"]
                                            answer_placeholder.markdown(final_answer)
                                            
                                elif event == "done":
                                    status_text.markdown("<div style='color: #10b981; font-weight: 600; padding: 8px 12px; background: rgba(16, 185, 129, 0.1); border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.2); width: fit-content; margin-bottom: 12px;'>✨ Pipeline Complete</div>", unsafe_allow_html=True)
                                    
                                elif event == "error":
                                    status_text.markdown(f"<div style='color: #ef4444; font-weight: 600; padding: 8px 12px; background: rgba(239, 68, 68, 0.1); border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.2); width: fit-content; margin-bottom: 12px;'>❌ Error: {data.get('message')}</div>", unsafe_allow_html=True)
                                    
                            except json.JSONDecodeError:
                                continue

        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to the API. Is your FastAPI server running on port 8000?")
        except requests.exceptions.Timeout:
            st.error("⚠️ Request timed out. The server may be overloaded. Please try again.")
        except requests.exceptions.HTTPError as e:
            st.error(f"⚠️ Server error: {e.response.status_code}. Please try again later.")
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ An unexpected error occurred. Please try again.")
            
    # Append the final generated text to the chat history
    if final_answer:
        st.session_state.messages.append({"role": "assistant", "content": final_answer})