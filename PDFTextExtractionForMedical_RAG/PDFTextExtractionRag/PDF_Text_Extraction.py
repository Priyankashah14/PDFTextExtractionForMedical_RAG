import os
import time
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from openai import OpenAI

#from langchain.chains import create_retrieval_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_core.prompts import ChatPromptTemplate

pdf_path = Path(r"C:\Users\priya\OneDrive\Documents\2017VavularRegurgitationGuideline.pdf")
chromaDB_path = Path(r"C:\Users\priya\OneDrive\Documents\AI Projects\PDFTextExtractionForMedical_RAG\chroma_db")
collection_name="medical_guideline"
Embedding_Model_Name = ("sentence-transformers/all-mpnet-base-v2")

# # TO DO : Write a code to check if the files exists otherwise create one"

start = time.perf_counter()

#load the PDF
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

print(
    f"PDF Load time: "
    f"in {time.perf_counter() - start:.2f} seconds."
    )

#split PDF into Chunks
start = time.perf_counter()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n","\n",". ", ","])
chunks = text_splitter.split_documents(docs) # coverting text into chunks

print(
    f"Chunks creation time: "
    f"in {time.perf_counter() - start:.2f} seconds."
    )

#Give every chunk a stable ID
chunk_ids = []
for index, chunk in enumerate(chunks):
    page = chunk.metadata.get("page","unknown",)
    chunk_ids.append(
        f"{pdf_path.stem}-page-{page}-chunk-{index}"
        )

#Load HuggingFaceEmbedding Model
start = time.perf_counter()
embedding_model = HuggingFaceEmbeddings(
    model_name=Embedding_Model_Name,
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 32,
    }
)

print(
    f"Embedded Model Load Time: "
    f"in {time.perf_counter() - start:.2f} seconds."
    )

# Embed all chunk and save them in chroma

start = time.perf_counter()
Chroma.from_documents(
    documents=chunks,
    ids = chunk_ids,
    embedding=embedding_model,
    collection_name=collection_name,
    persist_directory=str(chromaDB_path)
)

print(
    f"Total indexing Time: "
    f"in {time.perf_counter() - start:.2f} seconds."
    )


