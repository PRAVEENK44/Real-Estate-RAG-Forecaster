# RAG-Driven Predictive Analytics

This is a high-performance analytics platform that bridges the gap between unstructured document intelligence and structured statistical forecasting. By integrating Retrieval-Augmented Generation (RAG) with Facebook Prophet, this system provides a dual-engine approach to market analysis and asset valuation.

---

## 💠 Core Architecture

The system is bifurcated into two specialized engines:

1.  **The Intelligence Engine (RAG)**: Leverages **LangChain**, **FAISS**, and **Ollama (Llama 3.2)** to transform static PDFs into interactive knowledge bases. It uses neural embeddings (`all-MiniLM-L6-v2`) to perform semantic search and contextual reasoning.
2.  **The Forecasting Engine**: Implements **Facebook Prophet** for additive time-series modeling. It processes high-resolution Zillow Real Estate data to generate predictive horizons with statistical confidence intervals.

---

## 🛠 Tech Stack

*   **Frontend**: Streamlit (Premium UI Architecture)
*   **Vector Database**: FAISS (Facebook AI Similarity Search)
*   **LLM Integration**: LangChain & Ollama / OpenAI GPT-4o
*   **Embeddings**: HuggingFace Transformers
*   **Data Science**: Facebook Prophet, Pandas, Plotly, NumPy
*   **Document Analysis**: PyPDF & Unstructured Parsing

---

## 🚀 Deployment & Installation

### 1. Prerequisites
Ensure you have Python 3.10+ and [Ollama](https://ollama.com/) installed.

### 2. Environment Setup
Clone the repository and install the production-grade dependencies:

```bash
git clone https://github.com/[your-username]/nexus-insight.git
cd nexus-insight
pip install -r requirements.txt
```

### 3. Local LLM Configuration
Initialize the local model:
```bash
ollama run llama3.2
```

### 4. Application Launch
```bash
streamlit run app.py
```

---

## 📊 Business Logic & Use Cases

This system is designed for high-stakes environments where historical trends and specific institutional knowledge must coexist:
*   **Real Estate Investment Trusts (REITs)**: Analyzing market reports while forecasting property valuations.
*   **Supply Chain Intelligence**: Querying vendor contracts while predicting demand fluctuations.
*   **Financial Services**: Auditing compliance documents while modeling market volatility.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
