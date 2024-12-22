import streamlit as st
import random
import joblib
import plotly.express as px
import time


def load_files():
    st.sidebar.title("Upload Model Files")
    st.sidebar.markdown("🔄 **Upload the model and scaler files to get started.**")
    
    model_file = st.sidebar.file_uploader("Upload Model (.pkl)", type="pkl")
    scaler_file = st.sidebar.file_uploader("Upload Scaler (.pkl)", type="pkl")

    if model_file and scaler_file:
        try:
            with st.spinner("Loading files..."):
                model = joblib.load(model_file)
                scaler = joblib.load(scaler_file)
            st.sidebar.success("🎉 Model and scaler loaded successfully!")
            return model, scaler
        except Exception as e:
            st.sidebar.error(f"Error loading files: {e}")
            return None, None
    else:
        st.sidebar.info("📥 Please upload both model and scaler files.")
        return None, None


def predict_and_visualize(model, scaler, input_features):
    try:
        st.info("🧠 **Processing your input...**")
        input_array = [input_features]

        # Scale the inputs if a valid scaler is provided
        if scaler and hasattr(scaler, "transform"):
            input_scaled = scaler.transform(input_array)
        else:
            input_scaled = input_array

        # Simulate processing time for better user experience
        with st.spinner("🔍 Making predictions..."):
            time.sleep(1)
            prediction = model.predict(input_scaled)

        # Handle probabilities if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_scaled)[0]
            class_labels = model.classes_
        else:
            probabilities = None
            class_labels = None

        # Display prediction results
        st.success(f"✨ **Predicted Class:** `{prediction[0]}`")

        # Display class probabilities (if available)
        if probabilities is not None and class_labels is not None:
            st.markdown("### Class Probabilities")
            prob_fig = px.bar(
                x=class_labels,
                y=probabilities,
                labels={"x": "Class", "y": "Probability"},
                title="Class Probabilities",
                color_discrete_sequence=["#636EFA"],
            )
            st.plotly_chart(prob_fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")


def main():
    st.set_page_config(page_title="ML Model App", page_icon="🤖", layout="wide")
    st.title("🤖 ML Model Implementation")

    # Load model and scaler
    model, scaler = load_files()
    if not model or not scaler:
        return

    # Organize the UI into tabs
    tabs = st.tabs(["🎛️ Input Features", "📈 Prediction Results"])
    
    # Define feature names
    feature_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else ["length (mm)", "width (mm)", "density (g/cm³)", "pH"]
    input_features = []

    with tabs[0]:  # Input features tab
        st.header("Enter Feature Values")
        st.markdown("💡 Use the sliders below to provide input features for prediction.")

        for idx, feature in enumerate(feature_names):
            value = st.slider(
                label=feature,
                min_value=0.0,
                max_value=1000.0,
                value=st.session_state.get(f"input_{idx}", 0.0),
                step=0.1,
                key=f"input_{idx}"
            )
            input_features.append(value)

        # Button to make predictions
        if st.button("🚀 Predict"):
            st.session_state["make_prediction"] = True

    with tabs[1]:  # Prediction results tab
        st.header("Prediction Results")
        if st.session_state.get("make_prediction"):
            predict_and_visualize(model, scaler, input_features)
            st.session_state["make_prediction"] = False


if __name__ == "__main__":
    main()
