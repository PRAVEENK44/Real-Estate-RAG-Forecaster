"""
RAG-Driven Analytics: Intelligence Engine
Manages the Retrieval-Augmented Generation pipeline and LLM interactions.
"""

import os
import time
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from core.embeddings import EmbeddingsManager

class ChatbotManager:
    """Orchestrates the Retrieval-Augmented Generation (RAG) process."""
    
    def __init__(self, use_openai=False):
        self.embed_manager = EmbeddingsManager()
        self.use_openai = use_openai and os.getenv("OPENAI_API_KEY")
        self.llm = self._initialize_llm()
        self.qa_chain = None
        self.setup_qa_chain()

    def _initialize_llm(self):
        """Initializes either Ollama (local) or OpenAI based on configuration."""
        if self.use_openai:
            return ChatOpenAI(model="gpt-4o", temperature=0.3)
        else:
            return Ollama(
                model="llama3.2",
                temperature=0.3,
                num_ctx=4096
            )

    def setup_qa_chain(self, index_path="faiss_index"):
        """Configures the RetrievalQA chain."""
        try:
            if not os.path.exists(index_path):
                return
                
            vector_store = self.embed_manager.get_vector_store(index_path)
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": 5}
                ),
                return_source_documents=True
            )
        except Exception as e:
            print(f"QA Setup Error: {str(e)}")

    def get_response(self, query):
        """Generates a response using the RAG pipeline."""
        try:
            if not self.qa_chain:
                return "System not ready. Please verify document ingestion."
            
            formatted_query = f"""
            Context-Aware Query: {query}
            
            Instruction: Using only the provided documents, provide a detailed and professional response. 
            If the answer is not in the context, state that clearly.
            """
            
            response = self.qa_chain.invoke({"query": formatted_query})
            return response.get('result', "No response generated.")
            
        except Exception as e:
            return f"Intelligence Engine Error: {str(e)}"
