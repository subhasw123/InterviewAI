from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

docs = []

pdf_folder = "data"

for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        path = os.path.join(pdf_folder, file)

        print(f"Loading {file}")

        loader = PyPDFLoader(path)

        docs.extend(loader.load())

print(f"\nTotal pages loaded: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print(f"Total chunks: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(
    chunks,
    embeddings
)

db.save_local("vectorstore")

print("\nVectorstore created successfully")