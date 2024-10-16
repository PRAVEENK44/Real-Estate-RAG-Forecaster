import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv

# Import refactored core modules
from core.embeddings import EmbeddingsManager
from core.chatbot import ChatbotManager
from core.forecasting import RealEstatePredictor

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced RAG & Forecasting",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    /* Premium Look & Feel */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(#58A6FF, #2188FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #8B949E;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #161B22;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #30363D;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2EA043;
        box-shadow: 0 0 10px rgba(46, 160, 67, 0.4);
    }
    /* Hide specific Streamlit elements for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'embeddings_ready' not in st.session_state:
    st.session_state.embeddings_ready = False

# --- Sidebar ---
with st.sidebar:
    st.markdown("<div style='text-align: center;'><h1 style='color: #58A6FF;'>💠 ANALYTICS CORE</h1></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("Configuration")
    use_openai = st.toggle("Use OpenAI (GPT-4o)", value=False, help="Requires OPENAI_API_KEY in .env")
    
    st.markdown("---")
    st.subheader("Author")
    st.markdown("**Nexus AI Labs**")
    st.markdown("Directed by: [Your Name]") # USER will replace this or I will if I knew it
    
    if st.button("Clear Cache & Reset"):
        st.session_state.messages = []
        st.session_state.embeddings_ready = False
        st.rerun()

# --- Main Interface ---
st.markdown("<h1 class='main-header'>Predictive Analytics Studio</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Next-generation Retrieval-Augmented Generation & Asset Price Forecasting</p>", unsafe_allow_html=True)

# Grid Layout
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📂 Data Ingestion")
    uploaded_files = st.file_uploader(
        "Securely upload PDF documents or Zillow-format CSVs",
        type=["pdf", "csv"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        pdfs = [f for f in uploaded_files if f.name.endswith('.pdf')]
        csvs = [f for f in uploaded_files if f.name.endswith('.csv')]
        
        if pdfs:
            st.info(f"Detected {len(pdfs)} document(s)")
            if not st.session_state.embeddings_ready:
                if st.button("Process Neural Index"):
                    # Save temp files for processing
                    if not os.path.exists("temp_ingest"):
                        os.makedirs("temp_ingest")
                    
                    paths = []
                    for pdf in pdfs:
                        path = os.path.join("temp_ingest", pdf.name)
                        with open(path, "wb") as f:
                            f.write(pdf.getbuffer())
                        paths.append(path)
                    
                    manager = EmbeddingsManager()
                    count = manager.create_embeddings(paths)
                    st.session_state.embeddings_ready = True
                    st.session_state.chatbot = ChatbotManager(use_openai=use_openai)
                    st.success(f"Successfully indexed {count} context chunks.")
                    st.rerun()
            else:
                st.success("✅ Context Engine Active")

        if csvs:
            st.markdown("---")
            st.info(f"Detected {len(csvs)} dataset(s)")
            dataset = csvs[0] # Handle first one for prediction
            df = pd.read_csv(dataset)
            regions = sorted(df['RegionName'].unique())
            
            selected_region = st.selectbox("Select Target Region/Market", regions)
            horizon = st.slider("Forecast Horizon (Months)", 3, 36, 12)
            
            if st.button("Run Forecasting Engine"):
                with st.spinner("Executing Prophet Sequence..."):
                    predictor = RealEstatePredictor()
                    predictor.train(df, selected_region)
                    metrics = predictor.evaluate_performance()
                    
                    forecast, fig = predictor.generate_forecast(horizon)
                    
                    # Store results in session
                    st.session_state.forecast_data = {
                        'fig': fig,
                        'metrics': metrics,
                        'table': forecast
                    }
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    # --- Chat Interface ---
    st.header("💬 Intelligence Interface")
    if st.session_state.embeddings_ready:
        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Query the knowledge base..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    response = st.session_state.chatbot.get_response(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload and index documents to enable the Intelligence Interface.")

# --- Forecasting Results (Full Width Bottom) ---
if 'forecast_data' in st.session_state:
    st.markdown("---")
    st.header("📈 Predictive Analytics")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    metrics = st.session_state.forecast_data['metrics']
    m_col1.metric("Model RMSE", metrics['RMSE'])
    m_col2.metric("Mean Accuracy (MAPE)", metrics['MAPE'])
    m_col3.metric("System Confidence", metrics['Confidence'])
    
    st.plotly_chart(st.session_state.forecast_data['fig'], use_container_width=True)
    
    with st.expander("View Raw Forecast Data"):
        st.dataframe(st.session_state.forecast_data['table'], use_container_width=True)
