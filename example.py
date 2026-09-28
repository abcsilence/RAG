#import libraries
import os
import numpy as np
import faiss

from groq import Groq
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

#Setup Groq API
from getpass import getpass
api_key = getpass("Enter your Groq API key: ")

client = Groq(api_key=api_key)
Model = "openai/gpt-oss-20b"

#Ask LLM without rag
message = [{"role": "user", "content": "What is inside my private pdf ?"}]
response = client.chat.completions.create(
  model=Model,
  messages=message
)
print(response.choices[0].message.content)

#Load Embedding MOdel
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#Create Sample Documents
documents = [
    "RAG stand for Retrieval-Augmented Generation.",
    "Embeddings are vector representations of text that capture semantic meaning.",
    "Faiss is a library for efficient similarity search and clustering of dense vectors.",
    "Chunking is the process of breaking down large documents into smaller, manageable pieces for processing.",

]
print("Documents: ", documents)
    

#Generate Embiedding
embeddings = embedding_model.encode(documents)
print("Embeddings: ", embeddings)

#Create FAISS Index
dimension = embeddings.shape[1]
nlist = 2
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
index.train(embeddings)
index.add(embeddings)

#Ask a Question

query = "What does RAG stand for?"
query_embedding = embedding_model.encode([query])

#Search Similar Chunks 
distances, indices = index.search(np.array(query_embedding, dtype = "float32"), k=2)
print(indices)

#Retrive MAtching Documents
retrieved_docs = [documents[i] for i in indices[0]]
print("Retrieved Documents: ", retrieved_docs)


#Create Context
context = " ".join(retrieved_docs)
print("Context: ", context)

#Send Context to LLM
message = [{"role": "user", "content": f"Answer the following question based on the context provided: {query}. Context: {context}"}]
response = client.chat.completions.create(
    model=Model,
    messages=message
)
print(response.choices[0].message.content)


#Load pdf filre

#Chunk the pdf text

#