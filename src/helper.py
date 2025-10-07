# helper.py
# Updated for LangChain v0.2+ and huggingface embeddings

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------
# Extract Data From PDF Files
# ---------------------------
def load_pdf_file(data_path):
    """
    Loads all PDF files from the given directory.
    """
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# ---------------------------
# Split text into chunks
# ---------------------------
def text_split(extracted_data):
    """
    Splits documents into smaller chunks for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# ---------------------------
# Load HuggingFace Embeddings
# ---------------------------
def download_hugging_face_embeddings():
    """
    Returns HuggingFace embeddings object.
    Model: sentence-transformers/all-MiniLM-L6-v2 (384-dimensions)
    """
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings
