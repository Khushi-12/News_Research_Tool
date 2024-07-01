import os
import streamlit as st
import pickle
import time
import tqdm as notebook_tqdm
import langchain
import numpy as np
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains import RetrievalQAWithSourcesChain
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.docstore import InMemoryDocstore
from model import QuestionAnsweringAgent
import faiss
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

st.title(" News Research Tool")

st.sidebar.title("News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
    
file_path = 'vector_index.pkl'
button_clicked = st.sidebar.button("Process URL")
loading_bar = st.empty()
if button_clicked:
    #load data
    loader = UnstructuredURLLoader(urls = urls)
    data = loader.load()
    loading_bar.text("Data Loading has started")
    #split_text
    splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n","\n","."," "],
        chunk_size = 100,
        chunk_overlap = 5
    )

    docs = splitter.split_documents(data)
    loading_bar.text("Chunk Splitting has started")
    # print(docs)
    #create embeddings
    # if docs:
    embed_model = HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L6-v2')
    vector_index = FAISS.from_documents(docs, embed_model)
    loading_bar.text("Embeddings done")
    #save/open embeddings
    # if os.path.exists(file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(vector_index,f)
    

query = loading_bar.text_input("Question: ")
st.header(query)

if query:
    agent = QuestionAnsweringAgent()
    answer = agent.generate_answer(query)
    st.header("Answer")
    st.subheader(answer)

            






