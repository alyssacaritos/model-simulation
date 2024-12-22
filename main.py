import streamlit as st
import importlib.util

# Import the necessary files
from pages import app, LearningModel

st.set_page_config(page_title="Synthetic Data Generation", page_icon="♨️")

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
