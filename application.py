from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

# Hugging Face
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Pinecone setup
# ---------------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in environment variables")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ---------------------------
# Load embeddings and Pinecone index
# ---------------------------
embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ---------------------------
# Hugging Face LLM setup
# ---------------------------
generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    tokenizer="mistralai/Mistral-7B-Instruct-v0.2",
    max_new_tokens=500,
    temperature=0.4
)

llm = HuggingFacePipeline(pipeline=generator)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# ---------------------------
# RAG chain setup
# ---------------------------
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---------------------------
# Flask routes
# ---------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
