import streamlit as st
from rag import load_rag
from datetime import datetime
import uuid

# -------------------------------------------------
# MULTI-CHAT STATE
# -------------------------------------------------
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}

if "creating_chat" not in st.session_state:
    st.session_state.creating_chat = False

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat = chat_id
    st.session_state.chats[chat_id] = []

current_messages = st.session_state.chats[st.session_state.current_chat]


# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="NCERT AI Tutor for class 6-12",
    page_icon="📘",
    layout="wide"
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown("## 📘 NCERT AI Tutor")
    st.caption("Class 6–12 • RAG-based • Free AI")

    st.divider()

    st.markdown("### 💬 Chats")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.creating_chat = True

    if st.session_state.creating_chat:
        chat_name = st.text_input(
            "Enter chat name",
            placeholder="e.g. Photosynthesis – Class 7",
            key="new_chat_name"
        )

        if st.button("Create Chat", use_container_width=True):
            if chat_name.strip():
                chat_id = str(uuid.uuid4())
                st.session_state.chats[chat_id] = []
                st.session_state.chat_titles[chat_id] = chat_name.strip()
                st.session_state.current_chat = chat_id

                # JUST FLIP FLAG + RERUN
                st.session_state.creating_chat = False
                st.rerun()


    for chat_id in st.session_state.chats:
        title = st.session_state.chat_titles.get(chat_id, "Untitled Chat")
        if st.button(title, key=chat_id, use_container_width=True):
            st.session_state.current_chat = chat_id
            st.rerun()


    st.divider()

    st.markdown("### 📤 Export Chat")
    if current_messages:
        chat_text = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in current_messages
        )

        st.download_button(
            label="⬇ Download Chat (.txt)",
            data=chat_text,
            file_name=f"ncert_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.caption("No chat yet")

    st.divider()
    st.caption("Built for students • No ads")

# -------------------------------------------------
# Load RAG
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_rag():
    return load_rag()

rag_chain = init_rag()

current_chat_title = st.session_state.chat_titles.get(
    st.session_state.current_chat,
    "New Chat"
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    f"""
    <div class="chat-container">
        <h1>📚 {current_chat_title}</h1>
        <p style="color: gray;">
            Ask NCERT-based questions and get clear, exam-focused answers.
        </p>
        <hr>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Chat Container
# -------------------------------------------------
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Chat Input
# -------------------------------------------------
prompt = st.chat_input("Ask from NCERT textbooks...")

if prompt:
    current_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chat_history_list = [
        f"{m['role'].capitalize()}: {m['content']}"
        for m in current_messages[-6:]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching NCERT concepts..."):
            answer = rag_chain(
                {"question": prompt},
                chat_history_list
            )
            st.markdown(answer)

    current_messages.append({"role": "assistant", "content": answer})



# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; margin-top:2rem; color:gray; font-size:0.85rem;">
        NCERT AI Tutor • Retrieval-Augmented Generation • Free Models
    </div>
    """,
    unsafe_allow_html=True
)
