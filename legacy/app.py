import pickle

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# -----------------------
# Load Documents
# -----------------------

with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)
 
print("Documents Loaded")


# -----------------------
# Embeddings
# -----------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device":"cuda"},
    encode_kwargs={"normalize_embeddings":True}
)

print("Embeddings Loaded")


# -----------------------
# Load FAISS
# -----------------------

vectorstore = FAISS.load_local(
    "egypt_faiss",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)

print("FAISS Loaded")


# -----------------------
# Load Mistral
# -----------------------

llm = ChatOllama(
    model="mistral",
    temperature=0.3
)

print("Mistral Loaded")


# -----------------------
# Prompt
# -----------------------

prompt = """
You are an AI Egypt Tourism Assistant.

Use ONLY the following context.

Context:

{context}

Question:

{question}

Rules:

- Answer only from the context.
- If the answer is unavailable, say:
Not available in the knowledge base.
"""


# -----------------------
# Ask Function
# -----------------------

def ask_rag(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(final_prompt)

    return response.content


# -----------------------
# FastAPI
# -----------------------

app = FastAPI(
    title="Egypt AI Tourism Guide"
)


class Question(BaseModel):
    question: str


@app.get("/")
def home():

    return {
        "message": "Egypt AI Tourism Guide API"
    }


@app.post("/ask")
def ask(data: Question):

    answer = ask_rag(data.question)

    return {
        "question": data.question,
        "answer": answer
    }