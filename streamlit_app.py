import streamlit as st
import requests

# -----------------------
# Page Config
# -----------------------

st.set_page_config(
    page_title="Egypt AI Tourism Guide",
    page_icon="🇪🇬",
    layout="wide"
)

# -----------------------
# CSS
# -----------------------

st.markdown("""
<style>

.main{
    background-color:#f7f7f7;
}

.title{
    text-align:center;
    color:#1f3c88;
    font-size:45px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.answer-box{
    background-color:white;
    padding:20px;
    border-radius:12px;
    border-left:8px solid #d4af37;
    box-shadow:0px 0px 8px rgba(0,0,0,0.1);
}

.footer{
    text-align:center;
    color:gray;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# Sidebar
# -----------------------

with st.sidebar:

    st.title("🇪🇬 Egypt AI")

    st.write("---")

    st.subheader("About")

    st.write(
        """
        AI Tourism Assistant powered by:

        ✅ LangChain

        ✅ FAISS

        ✅ Ollama

        ✅ FastAPI

        ✅ Streamlit
        """
    )

    st.write("---")

    st.subheader("Example Questions")

    st.write("🏛 Who built Abu Simbel?")

    st.write("📍 Where is Luxor Temple?")

    st.write("🗺 Plan a one-day trip to Luxor.")

    st.write("🏺 Recommend ancient temples.")

# -----------------------
# Main Page
# -----------------------

st.markdown(
    "<div class='title'>🇪🇬 Egypt AI Tourism Guide</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Explore Egypt with AI-powered Tourism Assistant</div>",
    unsafe_allow_html=True
)

st.write("")

question = st.text_input(
    "💬 Ask your question",
    placeholder="Example: Who built Abu Simbel?"
)

col1, col2, col3 = st.columns([1,1,1])

with col2:

    ask = st.button(
        "🔍 Ask AI",
        use_container_width=True
    )

# -----------------------
# Ask API
# -----------------------

if ask:

    if question == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            try:

                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={
                        "question": question
                    }
                )

                if response.status_code == 200:

                    answer = response.json()["answer"]

                    st.write("")

                    st.subheader("🤖 AI Response")

                    st.markdown(
                        f"""
<div class="answer-box">

{answer}

</div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.error("FastAPI Error")

            except:

                st.error("Cannot connect to FastAPI.")

# -----------------------
# Divider
# -----------------------

st.write("")
st.divider()

# -----------------------
# Popular Questions
# -----------------------

st.subheader("🏛 Popular Questions")

c1, c2 = st.columns(2)

with c1:

    st.info("Who built Abu Simbel?")

    st.info("Where is Luxor Temple?")

    st.info("What are the opening hours of Abu Simbel?")

with c2:

    st.info("Plan a one-day trip to Luxor.")

    st.info("Recommend ancient temples.")

    st.info("Plan a two-day trip to Aswan.")

# -----------------------
# Footer
# -----------------------

st.write("")

st.markdown(
"""
<div class='footer'>

Made with ❤️ using

LangChain • FAISS • Ollama • FastAPI • Streamlit

</div>
""",
unsafe_allow_html=True
)