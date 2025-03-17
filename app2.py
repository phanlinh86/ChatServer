import streamlit as st
from openai import OpenAI

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
BASE_MODEL = "gemma3:12b"
PROMPT_ASSISTANT = ("The following is a conversation with an AI assistant.\n"
                    "The assistant is helpful, creative, clever, and very friendly.\n\n"
                    "User: Hello, who are you?\n"
                    "AI: I am an AI assistant. How can I help you today?\n"
                    "User: What can you do for me?\n"
                    "AI: I can help you with a variety of tasks. What do you need help with?\n"
                    "User:")

# Main app layout
st.title("AI & Media Web App")
st.header("Chat with AI")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = ""
    st.session_state['chat_messages'] = [{"role": "system", "content": PROMPT_ASSISTANT}]

# Chat history display
history_container = st.empty()
history_container.text(st.session_state['chat_history'])

# Initialize LLM client
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

# Chat input (submits on Enter)
chat_input = st.chat_input("Type your message...")

if chat_input:
    # Append user message to history
    st.session_state['chat_messages'].append({"role": "user", "content": chat_input})
    st.session_state['chat_history'] += "You: " + chat_input + "\n"
    history_container.text(st.session_state['chat_history'])

    # Get AI response
    response = client.chat.completions.create(
        model=BASE_MODEL,
        stream=True,
        messages=st.session_state['chat_messages']
    )
    ai_reply = ""
    st.session_state['chat_history'] += "AI: "
    for chunk in response:
        temp = chunk.choices[0].delta.content or ""
        ai_reply += temp
        st.session_state['chat_history'] += temp
        history_container.text(st.session_state['chat_history'])

    st.session_state['chat_history'] += "\n"
    st.session_state['chat_messages'].append({"role": "bot", "content": ai_reply})