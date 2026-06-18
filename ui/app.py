import streamlit as st
import requests
import json
import uuid

st.set_page_config(page_title="CRAG Agent", page_icon="🤖", layout="centered")

st.title("🔍 Corrective RAG Agent")
st.markdown("Ask a question and watch the LangGraph agent self-correct and reason in real-time.")

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
                                    status_text.caption(f"⚙️ **Agent is thinking...** currently running `[ {node} ]` node")
                                    
                                    # If we hit the generation or hallucination node, display the text
                                    if node in ["generator", "hallucination"]:
                                        if "generation" in updates:
                                            final_answer = updates["generation"]
                                            answer_placeholder.markdown(final_answer)
                                            
                                elif event == "done":
                                    status_text.caption("✅ **Pipeline Complete**")
                                    
                                elif event == "error":
                                    status_text.error(f"Error: {data.get('message')}")
                                    
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
    else:
        # Record failed exchanges so they persist in history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "⚠️ Failed to generate an answer. Please try again."
        })