from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import glob
from langchain_community.document_loaders.csv_loader import CSVLoader    
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader


CHROMA_PATH = "chroma"
DATA_PATH = "documents\\"

load_dotenv()


def generate_data_store():
    
    documents = load_documents()
    chunks = split_text(documents)
    return save_to_chroma(chunks)


def load_documents():

    documents = []

    # Load csv files
    for csv_file in glob.glob(os.path.join(DATA_PATH, "*.csv")):
        loader = CSVLoader(file_path=csv_file)
        doc = loader.load_and_split()
        documents.extend(doc)

    # Load xlsx files
    for xlsx_file in glob.glob(os.path.join(DATA_PATH, "*.xlsx")):
        loader = UnstructuredExcelLoader(file_path=xlsx_file)
        doc = loader.load_and_split()
        documents.extend(doc)

    # Load PDF files
    for pdf_file in glob.glob(os.path.join(DATA_PATH, "*.pdf")):
        loader = PyPDFLoader(pdf_file)
        pdf_documents = loader.load_and_split()
        documents.extend(pdf_documents)

    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks


def save_to_chroma(chunks: list[Document]):

    # Create a new DB
    embeddings = OpenAIEmbeddings()
    db = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PATH)

    # Add chunks in batches
    batch_size = 1000  # Adjust this batch size according to your needs and available memory
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        db.add_documents(batch_chunks)

    # Save the database
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    return db





