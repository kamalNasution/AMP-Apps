import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
import pickle
from PIL import Image
import joblib
import os
import time
import altair as alt

# Set page config with modern settings
st.set_page_config(
    page_title="Antibacterial Metabolite Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary: #4361ee;
        --secondary: #3f37c9;
        --accent: #4895ef;
        --success: #4cc9f0;
        --danger: #f72585;
        --warning: #f8961e;
        --light: #f8f9fa;
        --dark: #212529;
    }
    
    /* Main container */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    .welcome-header {
        font-size: 2.5rem !important;
        font-weight: 800;
        background: linear-gradient(45deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 0.5rem;
    }
    
    .subheader {
        font-size: 1.2rem !important;
        text-align: center;
        color: var(--dark);
        margin-bottom: 2rem;
        opacity: 0.8;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, var(--primary), var(--accent));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        margin: 10px 0;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(67, 97, 238, 0.2);
    }
    
    /* Cards */    
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        font-family: 'Segoe UI', sans-serif;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
        margin-top: 10px;
        border: 1px solid transparent;
    }

    .positive {
        background-color: #d4edda;  /* Hijau muda */
        color: #155724;
        border-color: #c3e6cb;
    }

    .negative {
        background-color: #f8d7da;  /* Merah muda */
        color: #721c24;
        border-color: #f5c6cb;
    }
    
    /* Tabs */
    .stTabs [role="tablist"] {
        justify-content: center;
        gap: 10px;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [role="tab"] {
        padding: 8px 20px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary);
        color: white;
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 10px 12px;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Spinner */
    .stSpinner>div {
        border-top-color: var(--primary);
    }
    
    /* Example SMILES */
    .example-smiles {
        font-family: monospace;
        background-color: #f0f2f6;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 5px 0;
        border: 1px solid #e0e0e0;
        transition: all 0.2s;
    }
    
    .example-smiles:hover {
        background-color: #e0e7ff;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Header with animation
header_col1, header_col2, header_col3 = st.columns([1, 3, 1])
with header_col2:
    st.markdown('<div class="welcome-header">🔬 AMP - Antibacterial Metabolite Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">AI-powered platform for discovering and analyzing antibacterial compounds in natural products</div>', unsafe_allow_html=True)
    
    # Usage instructions
    with st.expander("ℹ️ How to use this tool"):
        st.write("""
        **Structural Similarity**  
       Input your metabolite (SMILES), choose a similarity method, and click **"Find Similar Compounds"**.  
       The tool will show the **top 5 most similar compounds** from our curated database of natural antibiotics from **Jamu**, **Unani**, and **Traditional Chinese Medicine (TCM)**.

        **ML (Machine Learning) Prediction**  
       Input your metabolite (SMILES), choose a molecular descriptor and a machine learning model, then click **"Generate Prediction"**. Wait for the results to see if your compound has potential antibacterial activity.
        """)

@st.cache_data
def load_data():
    try:
        df = pd.read_excel('dataset.xlsx')
        df = df.dropna(subset=['Smiles'])
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_model(descriptor, model_type):
    model_map = {
        ('Morgan', 'Random Forest'): 'morgan_random_forest_best.joblib',
        ('Morgan', 'Neural Network'): 'morgan_neural_network_best.joblib',
        ('Morgan', 'SVM'): 'morgan_svm_best.joblib',
        ('MACCS', 'Random Forest'): 'maccs_random_forest_best.joblib',
        ('MACCS', 'Neural Network'): 'maccs_neural_network_best.joblib',
        ('MACCS', 'SVM'): 'maccs_svm_best.joblib'
    }
    model_filename = model_map.get((descriptor, model_type))
    if not model_filename:
        st.error("Model not found for the selected combination.")
        return None
    try:
        model_path = os.path.join("models", model_filename)
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def compute_descriptor_vector(smiles, descriptor_type="Morgan"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if descriptor_type.lower() == "morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    elif descriptor_type.lower() == "maccs":
        fp = MACCSkeys.GenMACCSKeys(mol)
    else:
        return None
    arr = np.zeros((1,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return pd.DataFrame([arr])

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptors = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
    }
    return descriptors

def calculate_similarities(query_smiles, target_smiles, method='tanimoto'):
    try:
        mol1 = Chem.MolFromSmiles(query_smiles)
        mol2 = Chem.MolFromSmiles(target_smiles)
        if mol1 is None or mol2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        similarity_methods = {
            'tanimoto': DataStructs.TanimotoSimilarity,
            'dice': DataStructs.DiceSimilarity,
            'cosine': DataStructs.CosineSimilarity,
            'sokal': DataStructs.SokalSimilarity,
            'russel': DataStructs.RusselSimilarity,
            'kulczynski': DataStructs.KulczynskiSimilarity,
            'mcconnaughey': DataStructs.McConnaugheySimilarity
        }
        method_func = similarity_methods.get(method.lower())
        return method_func(fp1, fp2) if method_func else 0.0
    except:
        return 0.0

def show_similarity_tab():
    # Header with explanation
    st.markdown("""
    ## 🔍 Structural Similarity Search
    *Find compounds structurally similar to known antibiotics using various similarity metrics.*
    """)
    
    # Organized input panel
    with st.container():
        st.markdown("### 🧪 Input your SMILES")
        col1, col2 = st.columns([2, 1])
        with col1:
            query_smiles = st.text_input(
                "Enter SMILES string:", 
                placeholder="C1=CC=CC=C1 (Benzene)", 
                key="similarity_input",
                help="Enter a valid SMILES string of your query compound"
            )
        with col2:
            method = st.selectbox(
                "Similarity Method",
                ["Tanimoto", "Dice", "Cosine", "Sokal", "Russel", "Kulczynski", "McConnaughey"],
                index=0,
                help="Select the similarity calculation method"
            )

    # Search button with visual effect
    search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
    with search_col2:
        if st.button("🔍 Find Similar Compounds", use_container_width=True, key="similarity_btn"):
            if not query_smiles or Chem.MolFromSmiles(query_smiles) is None:
                st.error("Please enter a valid SMILES string.")
            else:
                with st.spinner(f"Calculating {method} similarities..."):
                    time.sleep(2)
                    method_lower = method.lower()
                    df = load_data()
                    if not df.empty:
                        df['Similarity'] = df['Smiles'].apply(
                            lambda x: calculate_similarities(query_smiles, x, method_lower)
                        )
                        top_results = df.sort_values('Similarity', ascending=False).head(5)
                        
                        # Display results in attractive cards
                        st.markdown("### 📊 Top 5 Similar Compounds")
                        
                        for _, row in top_results.iterrows():
                            with st.container():
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    mol = Chem.MolFromSmiles(row['Smiles'])
                                    if mol:
                                        img = Draw.MolToImage(mol, size=(500, 500))
                                        st.image(img, use_container_width=True)
                                with col2:
                                    st.markdown(f"""
                                    **{row['Metabolite_name']}**  
                                    *Herbal medicine: {row['Herbal Medicine']}*

                                    **Antibiotics class**: {row['Antibiotics class']}  
                                    **Similarity ({method})**: `{row['Similarity']:.3f}`
                                    
                                    SMILES: `{row['Smiles']}`
                                    """)
                            
                            st.divider()

def show_ml_tab():
    # Header with explanation
    st.markdown("""
    ## 🤖 Machine Learning Prediction
    *Predict antibacterial properties using various molecular descriptors and machine learning models.*
    """)
    
    # Organized input panel
    with st.container():
        st.markdown("### 🧪 Input Parameters")
        smiles_input = st.text_input(
            "Enter SMILES string:",
            placeholder="C1=CC=CC=C1 (Benzene)",
            key="ml_input",
            help="Enter the SMILES string of the compound you want to analyze"
        )
        
        # Model parameter columns
        col1, col2 = st.columns(2)
        with col1:
            descriptor_option = st.selectbox(
                "Descriptor Type",
                ["Morgan", "MACCS"],
                key="desc_type",
                help="Choose the molecular descriptor type"
            )
        with col2:
            model_option = st.selectbox(
                "Model Type",
                ["Random Forest", "Neural Network", "SVM"],
                key="model_type",
                help="Select the machine learning model to use"
            )
    
    # Prediction button with visual effect
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        submit_btn = st.button(
            "🚀 Generate Prediction", 
            use_container_width=True, 
            key="predict_btn"
        )
    
    if submit_btn and smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if not mol:
            st.error("Invalid SMILES string. Please enter a valid structure.")
        else:
            with st.spinner("🧠 Predicting... Please wait."):
                time.sleep(2)
                input_df = compute_descriptor_vector(smiles_input, descriptor_option)
                if input_df is None:
                    st.error("Failed to compute descriptors. Try a different SMILES.")
                    return
                
                model = load_model(descriptor_option, model_option)
                if model is None:
                    st.error("Model could not be loaded.")
                    return
                
                try:
                    # Molecule visualization
                    st.markdown("### 🧬 Molecular Structure")
                    img = Draw.MolToImage(mol, size=(500, 500))
                    st.image(img, caption=f"Structure: {smiles_input}")
                    
                    # Prediction results
                    prediction = model.predict(input_df)
                    prediction_proba = model.predict_proba(input_df)
                    
                    predicted_class = prediction[0]
                    confidence = max(prediction_proba[0])
                    
                    # Display prediction in card
                    st.markdown("### 📊 Prediction Results")
                    #treshold cofidence
                    result_class = "positive" if confidence >= 0.75 else "negative"
                    st.markdown(
                        f"""
                        <div class="prediction-card {result_class}">
                            <h3>Predicted Class: {predicted_class}</h3>
                            <p>Confidence: {confidence:.2%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Probability visualization
                    st.markdown("### 📈 Class Probabilities")
                    proba_df = pd.DataFrame({
                        'Class': model.classes_,
                        'Probability': prediction_proba[0]
                    }).sort_values(by='Probability', ascending=False)
                    
                    # Create chart using Altair
                    chart = alt.Chart(proba_df).mark_bar().encode(
                        x='Probability:Q',
                        y=alt.Y('Class:N', sort='-x'),
                        color=alt.condition(
                            alt.datum.Class == predicted_class,
                            alt.value('#4361ee'),
                            alt.value('#cccccc')
                        ),
                        tooltip=['Class', 'Probability']
                    ).properties(height=300)
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# Tabs with improved styling
tab1, tab2 = st.tabs([
    "🔍 Structural Similarity", 
    "🤖 ML Prediction"

])

with tab1:
    show_similarity_tab()

with tab2:
    show_ml_tab()

# with tab3:
#     show_analysis_tab()

# Professional footer
st.markdown("---")
footer = """
<div style="text-align: center; padding: 20px; color: #6c757d; margin-top: 30px;">
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 10px;">
        <a href="#" style="color: #4361ee; text-decoration: none;">Documentation</a>
        <a href="#" style="color: #4361ee; text-decoration: none;">API</a>
        <a href="#" style="color: #4361ee; text-decoration: none;">GitHub</a>
        <a href="#" style="color: #4361ee; text-decoration: none;">Contact</a>
    </div>
    <div>
        Antibacterial Properties Discovery Tool
    </div>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)