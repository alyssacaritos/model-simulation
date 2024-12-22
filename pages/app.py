import streamlit as st
import random
import time
import os
import joblib
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


if "models" not in st.session_state:
    st.session_state["models"] = {}
if "results" not in st.session_state:
    st.session_state["results"] = {}

def load_data(file):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file)
        st.success("Data loaded successfully!")
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(data, target_column):
    """Preprocess the data by splitting it into features and target."""
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.success("Data preprocessing completed!")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None, None, None, None

def train_model(model, X_train, y_train):
    """Train the provided model with the training data."""
    try:
        model.fit(X_train, y_train)
        st.success("Model trained successfully!")
        return model
    except ValueError as ve:
        st.error(f"ValueError during training: {ve}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during training: {e}")
        return None

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model and display metrics."""
    try:
        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)
        st.text("Confusion Matrix:")
        st.text(cm)
        st.text("Classification Report:")
        st.text(report)
    except Exception as e:
        st.error(f"Error during evaluation: {e}")

def parse_input(input_string):
    """Parses a comma-separated input string into a list."""
    return [item.strip() for item in input_string.split(",")]

def initialize_class_dicts(features, classes):
    """Initializes or updates the mean and std dev dictionaries."""
    if "mean_values_dict" not in st.session_state:
        st.session_state.mean_values_dict = {}
    if "std_values_dict" not in st.session_state:
        st.session_state.std_values_dict = {}

    for class_name in classes:
        if class_name not in st.session_state.mean_values_dict:
            st.session_state.mean_values_dict[class_name] = [random.uniform(50, 150) for _ in features]
            st.session_state.std_values_dict[class_name] = [round(random.uniform(5.0, 15.0), 1) for _ in features]
        else:
            adjust_feature_count(class_name, features)

def adjust_feature_count(class_name, features):
    """Adjusts the feature count in existing dictionaries."""
    current_features = len(st.session_state.mean_values_dict[class_name])
    if current_features < len(features):
        for _ in range(len(features) - current_features):
            st.session_state.mean_values_dict[class_name].append(random.uniform(50, 150))
            st.session_state.std_values_dict[class_name].append(round(random.uniform(5.0, 15.0), 1))
    elif current_features > len(features):
        st.session_state.mean_values_dict[class_name] = st.session_state.mean_values_dict[class_name][:len(features)]
        st.session_state.std_values_dict[class_name] = st.session_state.std_values_dict[class_name][:len(features)]

def configure_class_settings(features, classes):
    """Configures per-class settings for mean and std dev values."""
    st.subheader("Class-Specific Settings")
    for class_name in classes:
        with st.expander(f"{class_name}"):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.mean_values_dict[class_name] = [
                    st.number_input(
                        f"Mean for {feature}",
                        value=st.session_state.mean_values_dict[class_name][i],
                        min_value=0.0,
                        step=0.1,
                        key=f"mean_{class_name}_{feature}"
                    ) for i, feature in enumerate(features)
                ]
            with col2:
                st.session_state.std_values_dict[class_name] = [
                    st.number_input(
                        f"Std Dev for {feature}",
                        value=st.session_state.std_values_dict[class_name][i],
                        min_value=0.1,
                        step=0.1,
                        key=f"std_{class_name}_{feature}"
                    ) for i, feature in enumerate(features)
                ]

def generate_synthetic_data(features, classes, total_sample_size):
    """Generates synthetic data for each class."""
    samples_per_class = total_sample_size // len(classes)
    remainder = total_sample_size % len(classes)
    class_data = []

    for i, class_name in enumerate(classes):
            extra_sample = 1 if i < remainder else 0
            num_samples = samples_per_class + extra_sample

            mean_values = st.session_state.mean_values_dict[class_name]
            std_values = st.session_state.std_values_dict[class_name]
            data = np.random.normal(
                loc=mean_values,
                scale=std_values,
                size=(num_samples, len(features))
            )
            class_labels = np.full((num_samples, 1), class_name)  # Class label column
            class_data.append(np.hstack([data, class_labels]))

    return class_data

def handle_data_output(features, classes, class_data, total_sample_size, train_test_split_percent):
    """Handles data processing and output display."""
    all_data = np.vstack(class_data)
    np.random.shuffle(all_data)

    train_size = train_test_split_percent / 100
    feature_data = all_data[:, :-1].astype(float)
    labels = all_data[:, -1]

    class_df = pd.DataFrame(feature_data, columns=features)
    class_df['Target'] = labels

    train_samples = int(train_size * total_sample_size)
    test_samples = total_sample_size - train_samples

    st.subheader("Dataset Split Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("Total Samples")
        st.subheader(total_sample_size)
    with col2:
        st.markdown("Training Samples")
        st.subheader(f"{test_samples} ({100 - train_test_split_percent}%)")
        
    with col3:
        st.markdown("Testing Samples")
        st.subheader(f"{train_samples} ({train_test_split_percent}%)")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(class_df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features)
    scaled_df['Target'] = labels

    st.subheader("Generated Data Sample")
    col1, col2 = st.columns([4, 4])
    with col1:
        st.write("Original Data (Random samples from each class):")
        st.dataframe(class_df, use_container_width=True)
    with col2:
        st.write("Scaled Data (using best model's scaler):")
        st.dataframe(scaled_df, use_container_width=True)

def sidebar_section():
    """Handles the sidebar UI and input collection."""
    st.header("Data Source")
    data_source = st.radio("Choose data source:", ["Generate Synthetic Data", "Upload Dataset"])

    features, classes = [], []
    total_sample_size, train_test_split_percent = 0, 0

    if data_source == "Generate Synthetic Data":
        st.subheader("Synthetic Data Generation")
        st.write("Define parameters for synthetic data generation below.")

        features = parse_input(st.text_input("Enter feature names (comma-separated)", "length (mm), width (mm), density (g/cm³)"))
        classes = parse_input(st.text_input("Enter class names (comma-separated)", "Ampalaya, Banana, Cabbage"))

        initialize_class_dicts(features, classes)
        configure_class_settings(features, classes)

        col1, col2 = st.columns(2)
        with col1:
            total_sample_size = st.slider("Number of samples", min_value=500, max_value=50000, step=500)
        with col2:
            train_test_split_percent = st.slider("Train-Test Split (%)", min_value=10, max_value=50, step=5)


    return data_source, features, classes, total_sample_size, train_test_split_percent


def load_and_prepare_data(class_data):
    all_data = np.vstack(class_data)
    np.random.shuffle(all_data)
    feature_data = all_data[:, :-1].astype(float)
    labels = all_data[:, -1]

    class_df = pd.DataFrame(feature_data, columns=[f"Feature_{i}" for i in range(feature_data.shape[1])])
    class_df['Target'] = labels
    return class_df


# --- FEATURE VISUALIZATION ---
def plot_2d_scatter(df, x_feature, y_feature):
    """
    Plots a 2D scatter plot with feature names displayed on the axes.

    Args:
        df: Pandas DataFrame containing the data.
        x_feature: Feature name for the x-axis (exact name from the DataFrame).
        y_feature: Feature name for the y-axis (exact name from the DataFrame).
    """
    fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color="Target",
        title=f"Scatter Plot of {x_feature} vs {y_feature}",
        labels={x_feature: x_feature, y_feature: y_feature},  # Display exact feature names
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_3d_scatter(df, x_feature, y_feature, z_feature):
    """
    Plots a 3D scatter plot with feature names displayed on the axes.

    Args:
        df: Pandas DataFrame containing the data.
        x_feature: Feature name for the x-axis (exact name from the DataFrame).
        y_feature: Feature name for the y-axis (exact name from the DataFrame).
        z_feature: Feature name for the z-axis (exact name from the DataFrame).
    """
    fig = px.scatter_3d(
        df,
        x=x_feature,
        y=y_feature,
        z=z_feature,
        color="Target",
        title=f"3D Scatter Plot of {x_feature}, {y_feature}, {z_feature}",
        labels={
            x_feature: x_feature,  
            y_feature: y_feature,  
            z_feature: z_feature   
        },
    )

    st.plotly_chart(fig, use_container_width=True)


def train_models(X_train, y_train, X_test, y_test):
    models = {
        "Gaussian Naive Bayes": GaussianNB(),
        "AdaBoost Classifier": AdaBoostClassifier(algorithm='SAMME'),
        "Random Forest Classifier": RandomForestClassifier(),
        "Support Vector Classification": SVC(),
        "Multi-layer Perceptron": MLPClassifier(max_iter=500),
        "Extra Trees Classifier": ExtraTreesClassifier(),
    }

    results = {}
    best_model = None
    best_score = 0

    for model_name, model in models.items():
        start_time = time.time()
        status = "Failed"

        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            status = "Success"

            results[model_name] = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "Training Time (s)": round(training_time, 4),
                "Status": status,
            }

            if accuracy > best_score:
                best_score = accuracy
                best_model = model
        except Exception as e:
            results[model_name] = {
                "Accuracy": None,
                "Precision": None,
                "Recall": None,
                "F1-Score": None,
                "Training Time (s)": None,
                "Status": status,
            }
            st.error(f"Error training {model_name}: {e}")

    return best_model, results, models


def display_classification_report(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("Classification Report (Best Model):")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)


def display_model_comparison(results):
    model_comparison_df = pd.DataFrame(results).T
    st.subheader("Model Comparison")
    st.dataframe(model_comparison_df)

def display_best_model_and_results(results):
   
    best_model_name = None
    best_accuracy = 0

    for model_name, metrics in results.items():
        if metrics["Accuracy"] is not None and metrics["Accuracy"] > best_accuracy:
            best_model_name = model_name
            best_accuracy = metrics["Accuracy"]

    # Display the best model
    if best_model_name:
        st.subheader("🚀 Best Model")
        st.write(f"{best_model_name}")
        st.write(f"\n Accuracy: **{best_accuracy * 100:.2f}%**")


def get_selected_models(model_results):
    """
    Allows users to select models for comparison.
    Returns a list of selected models.
    """
    selected_models = st.multiselect(
        "Select Models for Comparison", 
        options=list(model_results.keys()),  # List all available models
        default=list(model_results.keys())  # Pre-select all models
    )
    if not selected_models:
        st.warning("Please select at least one model to compare.")
    return selected_models

# Function to prepare metric data
def prepare_metric_data(model_results, selected_models, metrics):
    """
    Prepares metric data for selected models.
    Returns a DataFrame for plotting.
    """
    metric_values = {metric: [] for metric in metrics}
    for model_name in selected_models:
        results = model_results.get(model_name)
        if results and results['Status'] == 'Success':
            for metric in metrics:
                metric_values[metric].append(results.get(metric, 0))
        else:
            for metric in metrics:
                metric_values[metric].append(0)  # Default to 0 if no data
    return pd.DataFrame(metric_values, index=selected_models)

# Function to create and display the chart
def display_metrics_chart(metric_df, metrics):
    """
    Creates and displays a bar chart for performance metrics comparison.
    """
    fig = px.bar(
        metric_df,
        x=metric_df.index,  
        y=metrics,         
        title="Performance Metrics Comparison",
        labels={"value": "Score", "variable": "Metric", "index": "Model"},
        barmode='group',     
    )
    st.plotly_chart(fig, use_container_width=True)

# Main function to handle the performance metrics summary
def display_performance_summary(model_results):
    """
    Displays the performance metrics summary section.
    """
    st.subheader("Performance Metrics Summary")
    
    # Define metrics to compare
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Get selected models
    selected_models = get_selected_models(model_results)
    if not selected_models:
        return  
    
    # Prepare metric data
    metric_df = prepare_metric_data(model_results, selected_models, metrics)
    
    # Display metrics chart
    display_metrics_chart(metric_df, metrics)

def save_models(models, results, directory="saved_models"):
    os.makedirs(directory, exist_ok=True)
    for model_name, result in results.items():
        if result["Status"] == "Success":
            filepath = os.path.join(directory, f"{model_name}.pkl")
            joblib.dump(models[model_name], filepath)
    return directory

def convert_df_to_csv(df):
     return df.to_csv(index=False).encode('utf-8')

# Function to display the model accuracy table
def display_model_accuracy(results):
    model_accuracy_df = pd.DataFrame(
        [(model_name, results['Accuracy']) for model_name, results in results.items()],
        columns=["Model", "Accuracy"]
    )
    st.dataframe(model_accuracy_df)

# Function to save models
def save_models(models, results, saved_models_dir):
    if models and results:
        st.session_state["models"] = models
        st.session_state["results"] = results
        
        saved_models_dir = "saved_models"
        os.makedirs(saved_models_dir, exist_ok=True)

        for model_name, model in models.items():
            model_accuracy = results.get(model_name, {}).get("Accuracy", "N/A")
            if model_accuracy != "N/A":
                model_file_path = os.path.join(saved_models_dir, f"{model_name}.pkl")
                joblib.dump(model, model_file_path)
                st.session_state["saved_models"] = saved_models_dir
    else:
        st.error("No models or results found to save.")
# Function to display the download button for models
def display_download_button(saved_models_dir, model_accuracy_df):
    selected_model = st.selectbox("Select Model to Download", options=model_accuracy_df["Model"])

    if selected_model:
        model_file_path = os.path.join(saved_models_dir, f"{selected_model}.pkl")
        if os.path.exists(model_file_path):
            with open(model_file_path, "rb") as file:
                st.download_button(
                    label=f"Download {selected_model} (.pkl)",
                    data=file,
                    file_name=f"{selected_model}.pkl",
                    mime="application/octet-stream",
                )
        else:
            st.error(f"Model file for {selected_model} not found!")
# Function to plot the learning curve
def plot_learning_curve(estimator, title, X, y, cv=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure(figsize=(4, 4))
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='accuracy', n_jobs=-1
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt

# Function to display learning curves
def display_learning_curves(models, model_results, X_train, y_train):
    st.subheader("Learning Curves for All Models")

    model_names = list(models.keys())
    n_models = len(model_names)
    cols_per_row = 4
    rows = (n_models // cols_per_row) + (1 if n_models % cols_per_row else 0)

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            model_idx = row * cols_per_row + col_idx
            if model_idx < n_models:
                model_name = model_names[model_idx]
                model = models[model_name]
                model_accuracy = model_results.get(model_name, {}).get("Accuracy", "N/A")

                with cols[col_idx]:
                    fig = plot_learning_curve(
                        model,
                        f"{model_name} \n Accuracy: {model_accuracy:.2%})",
                        X_train,
                        y_train,
                        cv=5
                    )
                    st.pyplot(fig)

def plot_confusion_matrix(model, X_test, y_test, model_name, class_names, model_accuracy):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(f"{model_name} \n  Accuracy: {model_accuracy:.2%}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    
    return fig

# Function to display confusion matrices
def display_confusion_matrices(models, model_results, X_test, y_test):
    st.subheader("Confusion Matrix for Each Model")

    n_models = len(models)
    rows = (n_models + 2) // 4
    cols_per_row = 4
    model_names = list(models.keys())

    class_names = sorted(y_test.unique())

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            model_idx = row * cols_per_row + col_idx
            if model_idx < n_models:
                model_name = model_names[model_idx]
                model = models[model_name]

                # Fetch accuracy from model_results
                model_accuracy = model_results.get(model_name, {}).get("Accuracy", "N/A")

                with cols[col_idx]:
                    if model_results.get(model_name, {}).get("Status") == "Success":
                        fig = plot_confusion_matrix(
                            model, X_test, y_test, model_name, class_names, model_accuracy
                        )
                        st.pyplot(fig)
                    else:
                        st.warning(f"{model_name} did not train successfully.")
    
def main():
      
    st.title("ML Model Generator")
   
   
    # Initialize session state attributes
    if "mean_values_dict" not in st.session_state:
        st.session_state.mean_values_dict = {}
    if "std_values_dict" not in st.session_state:
        st.session_state.std_values_dict = {}

    with st.sidebar:
        
        data_source, features, classes, total_sample_size, train_test_split_percent = sidebar_section()
        generate_data_button = st.button("Generate Data and Train Model")

    if data_source == "Generate Synthetic Data":
        class_data = generate_synthetic_data(features, classes, total_sample_size)
        if generate_data_button or 'generated':
            handle_data_output(features, classes, class_data, total_sample_size, train_test_split_percent)
            
            all_data = np.vstack(class_data)
            np.random.shuffle(all_data)
            feature_data = all_data[:, :-1].astype(float)
            labels = all_data[:, -1]

            

            # Create DataFrame for class data
            class_df = pd.DataFrame(feature_data, columns=features)
            class_df['Target'] = labels

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(class_df[features]) 
            scaled_df = pd.DataFrame(scaled_data, columns=features)
            scaled_df['Target'] = labels  

            st.subheader("Feature Visualization")
            features = class_df.columns[:-1]  # Exclude 'Target' for plotting

            # Convert all features to numeric, coercing errors
            for feature in features:
                class_df[feature] = pd.to_numeric(class_df[feature], errors='coerce')

            # List of unique class labels
            classes = class_df['Target'].unique()

            features = list(features) if isinstance(features, pd.Index) else features

            # Initialize session state for features
            if "x_feature" not in st.session_state:
                st.session_state.x_feature = features[0]
            if "y_feature" not in st.session_state:
                st.session_state.y_feature = features[1] if len(features) > 1 else features[0]
            if "z_feature" not in st.session_state:
                st.session_state.z_feature = features[2] if len(features) > 2 else features[0]

            # Select visualization type
            visualization_type = st.radio("Select Visualization Type", ["2D", "3D"])

            if visualization_type == "2D":
                # Dropdowns for X and Y axes
                col1, col2 = st.columns(2)
                with col1:
                    x_feature = st.selectbox(
                        "Select X-Axis Feature",
                        features,
                        index=features.index(st.session_state.x_feature) if st.session_state.x_feature in features else 0,
                        key="x_feature_select"
                    )
                with col2:
                    y_feature = st.selectbox(
                        "Select Y-Axis Feature",
                        features,
                        index=features.index(st.session_state.y_feature) if st.session_state.y_feature in features else 0,
                        key="y_feature_select"
                    )
                plot_2d_scatter(class_df, x_feature, y_feature)


            elif visualization_type == "3D":
                # Dropdowns for X, Y, and Z axes
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_feature = st.selectbox(
                        "Select X-Axis Feature",
                        features,
                        index=features.index(st.session_state.x_feature) if st.session_state.x_feature in features else 0,
                        key="x_3d"
                    )
                with col2:
                    y_feature = st.selectbox(
                        "Select Y-Axis Feature",
                        features,
                        index=features.index(st.session_state.y_feature) if st.session_state.y_feature in features else 0,
                        key="y_3d"
                    )
                with col3:
                    z_feature = st.selectbox(
                        "Select Z-Axis Feature",
                        features,
                        index=features.index(st.session_state.z_feature) if st.session_state.z_feature in features else 0,
                        key="z_3d"
                    )
                plot_3d_scatter(class_df, x_feature, y_feature, z_feature)

            
            # Split data
            #train_test_split_percent = st.slider("Train/Test Split (%)", 10, 90, 80)
            X = class_df[features]
            y = class_df["Target"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_test_split_percent) / 100)
            
            st.subheader("Download Dataset")

            original_csv = convert_df_to_csv(class_df)
            scaled_csv = convert_df_to_csv(scaled_df)

            # Create download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Original Dataset (CSV)",
                    data=original_csv,
                    file_name="original_dataset.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    label="Scaled Dataset (CSV)",
                    data=scaled_csv,
                    file_name="scaled_dataset.csv",
                    mime="text/csv"
                )

            
            with st.expander("Dataset Statistics"):
                st.subheader("Dataset Statistics Overview")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Original Dataset**")
                    st.dataframe(class_df.describe())
                with col2:
                    st.write("**Scaled Dataset**")
                    st.dataframe(scaled_df.describe())
                    

            # Train models
            best_model, results, models = train_models(X_train, y_train, X_test, y_test)

            # Display results
            
            if best_model:
                display_best_model_and_results(results)
                
                display_classification_report(best_model, X_test, y_test)
                display_model_comparison(results)
                display_performance_summary(results)

                st.subheader("Saved Models")
                display_model_accuracy(results)
    
                saved_models_dir = "saved_models"
                save_models(models, results, saved_models_dir)

                model_accuracy_df = pd.DataFrame(
                    [(model_name, results['Accuracy']) for model_name, results in results.items()],
                    columns=["Model", "Accuracy"]
                )
                display_download_button(saved_models_dir, model_accuracy_df)
                
                display_learning_curves(models, results, X_train, y_train)
                
                display_confusion_matrices(models, results, X_test, y_test)


    elif data_source == "Upload Dataset":

        file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

        if file is not None:
            data = load_data(file)
            if data is not None:
                column_names = data.columns.tolist()
                target_column = st.selectbox("Select the target column name:", column_names)
                if target_column:
                    X_train, X_test, y_train, y_test = preprocess_data(data, target_column)
                    if X_train is not None and y_train is not None:

                        best_model, results, models = train_models(X_train, y_train, X_test, y_test)

                        if best_model:
                            display_best_model_and_results(results)
                            
                            display_classification_report(best_model, X_test, y_test)
                            display_model_comparison(results)
                            display_performance_summary(results)
    
            









    

if __name__ == "__main__":
    main()