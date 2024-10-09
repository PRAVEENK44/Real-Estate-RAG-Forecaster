"""
RAG-Driven Analytics: Embedding Management System
Handles document preprocessing, vectorization, and neural database persistence.
"""

import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class EmbeddingsManager:
    """Manages the creation and retrieval of vector embeddings for document analysis."""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "],
            length_function=len
        )

    def create_embeddings(self, pdf_files, index_path="faiss_index"):
        """
        Processes a list of PDF files and generates a FAISS vector index.
        """
        try:
            progress_bar = st.progress(0, text="Initializing engine...")
            
            # Step 1: Document Loading
            all_chunks = []
            for i, pdf_path in enumerate(pdf_files):
                progress_val = (i + 1) / len(pdf_files) * 0.4
                progress_bar.progress(progress_val, text=f"Analyzing document {i+1}/{len(pdf_files)}...")
                
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                chunks = self.text_splitter.split_documents(pages)
                all_chunks.extend(chunks)
            
            # Step 2: Vector Generation
            progress_bar.progress(0.6, text=f"Generating embeddings for {len(all_chunks)} neural chunks...")
            vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
            
            # Step 3: Persistence
            progress_bar.progress(0.9, text="Finalizing vector database...")
            vectorstore.save_local(index_path)
            
            progress_bar.progress(1.0, text="✅ Neural Index Ready")
            return len(all_chunks)
            
        except Exception as e:
            st.error(f"Embedding Error: {str(e)}")
            raise e

    def get_vector_store(self, index_path="faiss_index"):
        """Retrieves the persisted vector store."""
        return FAISS.load_local(
            index_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
