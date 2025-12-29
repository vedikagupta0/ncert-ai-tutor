import streamlit as st
from rag import load_rag
from datetime import datetime

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

    st.markdown("### 📤 Export Chat")
    if "messages" in st.session_state and st.session_state.messages:
        chat_text = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in st.session_state.messages
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
# Header
# -------------------------------------------------
st.markdown(
    """
    <div class="chat-container">
        <h1>📚 NCERT Class 6–12 AI Tutor</h1>
        <p style="color: gray;">
            Ask NCERT-based questions and get clear, exam-focused answers.
        </p>
        <hr>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Load RAG (Cached)
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_rag():
    return load_rag()

rag_chain = init_rag()

# -------------------------------------------------
# Chat State
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# Chat Container
# -------------------------------------------------
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Chat Input
# -------------------------------------------------
prompt = st.chat_input("Ask from NCERT textbooks...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching NCERT concepts..."):

            chat_history_list = [
                f"{m['role'].capitalize()}: {m['content']}"
                for m in st.session_state.messages
            ]

            answer, updated_history = rag_chain(
                {
                    "question": prompt,
                    "chat_history": chat_history_list
                },
                chat_history_list
            )

            st.markdown(answer)

    # Rebuild messages state from updated history
    st.session_state.messages = []
    for line in updated_history:
        if line.startswith("User: "):
            st.session_state.messages.append({"role": "user", "content": line[len("User: "):]})
        elif line.startswith("AI: "):
            st.session_state.messages.append({"role": "assistant", "content": line[len("AI: "):]})

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
