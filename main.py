import streamlit as st
import importlib.util

# Import the necessary files
from pages import app, LearningModel

st.set_page_config(page_title="Synthetic Data Generation", page_icon="🗃️")

def execute_py_file(file_path):
    """Execute a Python file by importing it dynamically."""
    spec = importlib.util.spec_from_file_location("module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

# Mapping files to their corresponding functions
file_functions = {
    "app.py": (app.main, "app"), 
    "LearningModel.py": (LearningModel.run, "Ed"),   
}

# App Title
st.title("📊 **Modeling and Simulation with Python**")

# Checkbox for Text-Only Mode
show_text_only = st.checkbox("View Text Only")

if show_text_only:

    st.write("""
    🌟 **Modeling and Simulation with Python** 🌟
             
             **📝Author**: 
            
            Caritos, Alyssa P.
            Dimanarig, Shiella R.
            Torallo, Gracia P.

    Welcome to this exciting project where we delve into the fascinating world of **modeling and simulation** using Python!  
    This hands-on project provides an opportunity to explore essential concepts, tools, and techniques that bring data-driven systems to life.  

    ### 🔍 **What’s This Project About?**  
    This project is all about **building models** and **running simulations** using Python. By generating synthetic data and leveraging Python’s rich ecosystem of libraries, we will:  
    - 🛠️ Explore how to analyze and simulate real-world scenarios.  
    - 📊 Build statistical models and evaluate their behavior.  
    - 🔄 Apply these techniques to gain deeper insights into systems and processes.  
    """)
else:
    # Introduction Section
    st.header("🔍 **What’s This Project About?**")
    st.markdown("""
    This project is all about **building models** and **running simulations** using Python.  
    By generating synthetic data and leveraging Python’s rich libraries, we will:  
    - 🛠️ Explore how to analyze and simulate real-world scenarios.  
    - 📊 Build statistical models and evaluate their behavior.   
    """)

    st.header("🛤️ **Project Roadmap**")
    st.markdown("""
    ### 1️⃣ **Introduction**  
    - Overview of Python libraries and tools for modeling and simulation.

    ### 2️⃣ **Project Overview**  
    - Key steps: **Data Generation, Exploratory Data Analysis, Modeling, Simulation, Evaluation, and Analysis**.

    ### 3️⃣ **Data Generation**  
    - Create synthetic data with **NumPy** and **scikit-learn**.  
    - Simulate real-world scenarios by designing distributions and correlations.

    ### 4️⃣ **Exploratory Data Analysis (EDA)**  
    - Use **pandas**, **matplotlib**, and **seaborn** for:  
      - Exploring statistical properties.  
      - Visualizing correlations and trends.
    """)
    ### 5️⃣ **Simulation**  

   # Additional Engagement: Why it Matters
    st.header("📚 **Why Modeling and Simulation Matter**")
    st.markdown("""
    Modeling and simulation are powerful tools for predicting outcomes, optimizing systems, and making data-driven decisions.  
    Applications span various fields:  
    - **Engineering**: Simulating mechanical systems.  
    - **Finance**: Risk prediction and optimization.  
    - **Healthcare**: Simulating disease spread or patient outcomes.  
    """)

