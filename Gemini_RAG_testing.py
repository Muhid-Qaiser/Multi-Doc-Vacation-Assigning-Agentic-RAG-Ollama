import os
import glob
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders.csv_loader import CSVLoader    
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.prompts import ChatPromptTemplate
import ollama


PROMPT_TEMPLATE = """
You are an AI assistant which provides detailed and long answers.

### IMPORTANT: Make sure your responses are detailed and long to provide the user with as much information as possible.

Answer the question based only on the given context.

Question : {question}

Conext : {context}

"""

CHROMA_PATH = "chroma"
DATA_PATH = "documents\\"

class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "mxbai-embed-large"):
        self.model = model
        self.client = ollama.Client()
    
    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings(model=self.model, prompt=text)
        embedding = response["embedding"]
        # Normalize the embedding vector using L2 normalization
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        normalized_embedding = [x / norm for x in embedding]
        return normalized_embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    return save_to_chroma(chunks)

def load_documents():
    documents = []
    # Load CSV files
    for csv_file in glob.glob(os.path.join(DATA_PATH, "*.csv")):
        loader = CSVLoader(file_path=csv_file)
        docs = loader.load_and_split()
        documents.extend(docs)
    # Load XLSX files
    for xlsx_file in glob.glob(os.path.join(DATA_PATH, "*.xlsx")):
        loader = UnstructuredExcelLoader(file_path=xlsx_file)
        docs = loader.load_and_split()
        documents.extend(docs)
    # Load PDF files
    for pdf_file in glob.glob(os.path.join(DATA_PATH, "*.pdf")):
        loader = PyPDFLoader(pdf_file)
        docs = loader.load_and_split()
        documents.extend(docs)
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Use OllamaEmbeddings instead of OpenAIEmbeddings
    embeddings = OllamaEmbeddings()  # uses default model "mxbai-embed-large"
    db = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PATH)
    
    # Add chunks in batches
    batch_size = 1000  # Adjust this batch size as needed
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        db.add_documents(batch_chunks)
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    return db

def predict(model, query_text, db):
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    
    if len(results) == 0 or results[0][1] < 0.2:
        return "Unable to find matching results."

    context_text = "\n---\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    response = ollama.generate(model=model_name, prompt=prompt)
    # answer = response.get("message", {}).get("content", "")
    answer = response['response']
    return answer

# Example usage
model_name = 'mistral:latest'
db = generate_data_store()

print("Chatbot: Hello! How can I help you today? (Type 'exit' to end the chat)")
while True:
    query_text = input("You: ")
    if query_text.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = predict(model_name, query_text, db)
    print(f"Chatbot: {response}")
