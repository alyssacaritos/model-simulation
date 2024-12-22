import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*ScriptRunContext.*")

def run():
# Streamlit App Title
    st.title("📚 Machine Learning Algorithms Guide 📊")
    st.write(
        "Welcome to the **Machine Learning Algorithms Guide**! 🎓"
        " Here, you can explore detailed information about various machine learning algorithms. "
        "Select an algorithm to dive deep into its workings, advantages, disadvantages, and more! 🤖"
    )

    st.write("Select an algorithm to learn more: 🧠")

    # User selects algorithm
    algorithm = st.selectbox(
        "Select a machine learning algorithm:",
        [
            "Gaussian Naive Bayes", "Multinomial Naive Bayes", "AdaBoost Classifier", 
            "Random Forest Classifier", "Support Vector Classification", "Multi-layer Perceptron", 
            "Extra Trees Classifier"
        ]
    )
    # Algorithm Details Section
    if algorithm == "Gaussian Naive Bayes":
        st.header("📊 Gaussian Naive Bayes (GNB) Algorithm")
        st.subheader("🔢 Mathematical Formula:")
        st.latex(r"P(y|X) = \frac{P(X|y) P(y)}{P(X)}")
        col1, col2 = st.columns(2)
    
        with col1:
            st.subheader("✅ Advantages:")
            st.write("- Simple and easy to implement.")
            st.write("- Fast training and prediction.")
            st.write("- Works well with categorical features.")
            st.write("- Great for text classification tasks (e.g., spam detection).")
    

        with col2:
    
            st.subheader("❌ Disadvantages:")
            st.write("- Assumes that features are independent, which may not hold in real-world data.")
            st.write("- Sensitive to irrelevant features and outliers.")
            st.write("- Doesn't perform well on correlated features.")
    
        st.subheader("💡 Use Cases:")
        st.write("- Email spam detection 📨")
        st.write("- Document classification 📄")
        st.write("- Sentiment analysis on text data 📈")
    
        st.subheader("📄 Example:")
        st.write("Check out how Gaussian Naive Bayes can be applied to a text classification task!")

    elif algorithm == "Multinomial Naive Bayes":
        st.header("📊 Multinomial Naive Bayes Algorithm")
        st.subheader("🔢 Mathematical Formula:")
        st.latex(r"P(y|X) = \frac{P(X|y) P(y)}{P(X)}")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Advantages:")
            st.write("- Works well for text classification tasks, especially for large feature spaces.")
            st.write("- Efficient when dealing with high-dimensional data.")
            st.write("- Handles class-imbalanced data well.")
        with col2:
            st.subheader("❌ Disadvantages:")
            st.write("- Assumes features are conditionally independent.")
            st.write("- Doesn't perform well with highly correlated features.")
    
        st.subheader("💡 Use Cases:")
        st.write("- Text classification 📝")
        st.write("- Spam email detection 📧")
        st.write("- Document categorization 📑")
    
        st.subheader("📄 Example:")
        st.write("Multinomial Naive Bayes is often used in document classification, where the feature space is high-dimensional.")

    elif algorithm == "AdaBoost Classifier":
        st.header("📊 AdaBoost Classifier Algorithm")
        st.subheader("🔢 Mathematical Formula:")
        st.latex(r"y = \sum_{t=1}^{T} \alpha_t h_t(x)")
        col1, col2 = st.columns(2)

        with col1:

            st.subheader("✅ Advantages:")
            st.write("- Combines multiple weak classifiers to create a strong classifier.")
            st.write("- Reduces bias and variance, improving model performance.")
            st.write("- Works well with many types of base classifiers.")
    
        with col2:
            st.subheader("❌ Disadvantages:")
            st.write("- Sensitive to noisy data and outliers.")
            st.write("- May overfit if too many iterations are used.")
    
        st.subheader("💡 Use Cases:")
        st.write("- Face detection in images 🖼️")
        st.write("- Medical diagnostics 🏥")
        st.write("- Text classification 📑")
    
        st.subheader("📄 Example:")
        st.write("AdaBoost is often used in image classification tasks where boosting weak learners improves performance.")

    elif algorithm == "Random Forest Classifier":
        st.header("📊 Random Forest Classifier Algorithm")
        st.subheader("🔢 Mathematical Formula:")
        st.latex(r"y = \frac{1}{T} \sum_{t=1}^{T} h_t(x)")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Advantages:")
            st.write("- Combines multiple decision trees to make robust predictions.")
            st.write("- Handles both classification and regression tasks.")
            st.write("- Handles large datasets with higher dimensionality well.")
            st.write("- Reduces the risk of overfitting compared to individual decision trees.")
        with col2:
            st.subheader("❌ Disadvantages:")
            st.write("- Can be computationally expensive, especially for large datasets.")
            st.write("- Model interpretability is lower than a single decision tree.")
    
        st.subheader("💡 Use Cases:")
        st.write("- Predicting customer churn 🔄")
        st.write("- Credit scoring 💳")
        st.write("- Feature selection in high-dimensional data 🧠")
    
        st.subheader("📄 Example:")
        st.write("Random Forest is used for tasks like predicting customer behavior or diagnosing medical conditions.")

    elif algorithm == "Support Vector Classification":
        st.header("📊 Support Vector Classification (SVC) Algorithm")
        st.subheader("🔢 Mathematical Formula:")
        st.latex(r"f(x) = w^T x + b")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Advantages:")
            st.write("- Effective in high-dimensional spaces.")
            st.write("- Works well with clear margin of separation.")
            st.write("- Robust to overfitting, especially in high-dimensional space.")
        with col2:
            st.subheader("❌ Disadvantages:")
            st.write("- Memory-intensive, especially for large datasets.")
            st.write("- Choosing the right kernel can be challenging.")
            st.write("- Does not perform well when data is noisy or not linearly separable.")
    
        st.subheader("💡 Use Cases:")
        st.write("- Image classification 🖼️")
        st.write("- Text classification 📝")
        st.write("- Bioinformatics (e.g., protein classification) 🧬")
    
        st.subheader("📄 Example:")
        st.write("SVC can be used to classify images or texts into different categories.")

    elif algorithm == "Multi-layer Perceptron":
        st.header("📊 Multi-layer Perceptron Algorithm")
        st.subheader("🔢 Mathematical Formula:")
        st.latex(r"y = \sigma(Wx + b)")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ Advantages:")
            st.write("- Can model complex relationships between input and output.")
            st.write("- Suitable for both classification and regression tasks.")
            st.write("- Performs well on large datasets with many features.")
        with col2:
            st.subheader("❌ Disadvantages:")
            st.write("- Requires a lot of computational power and training data.")
            st.write("- Prone to overfitting if not regularized properly.")
            st.write("- Difficult to interpret results.")
    
        st.subheader("💡 Use Cases:")
        st.write("- Speech recognition 🗣️")
        st.write("- Image recognition 🖼️")
        st.write("- Time-series forecasting ⏳")
    
        st.subheader("📄 Example:")
        st.write("MLPs are often used in deep learning tasks, such as image classification.")

    elif algorithm == "Extra Trees Classifier":
        st.header("📊 Extra Trees Classifier Algorithm")
        st.subheader("🔢 Mathematical Formula:")
        st.latex(r"y = \frac{1}{T} \sum_{t=1}^{T} h_t(x)")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ Advantages:")
            st.write("- Reduces variance by averaging multiple decision trees.")
            st.write("- Faster to train than Random Forests.")
            st.write("- Performs well on both classification and regression tasks.")
        with col2:
            st.subheader("❌ Disadvantages:")
            st.write("- Prone to overfitting with too many trees.")
            st.write("- Can be computationally expensive with large datasets.")
    
        st.subheader("💡 Use Cases:")
        st.write("- Predicting customer behavior 🔄")
        st.write("- Anomaly detection 🚨")
        st.write("- Time-series forecasting ⏳")
    
        st.subheader("📄 Example:")
        st.write("Extra Trees are used for ensemble learning tasks, where multiple trees are used for better predictions.")

    # Interactive Model Demo Section
    st.header("💻 Interactive Demo 🖱️")
    st.write("Test the selected model with different datasets and see how it performs. 🚀")

    # Dataset Selection
    dataset_option = st.selectbox(
        "Select a dataset for the demo: 📑", ["Iris Dataset", "Wine Dataset", "Digits Dataset"]
    )

# Generate data based on the dataset optio
    if dataset_option == "Iris Dataset":
        iris = load_iris(as_frame=True)
        data = iris.frame
        data.rename(columns={"target": "Label"}, inplace=True)

    elif dataset_option == "Wine Dataset":
        wine = load_wine(as_frame=True)
        data = wine.frame
        data.rename(columns={"target": "Label"}, inplace=True)

    elif dataset_option == "Digits Dataset":
        digits = load_digits(as_frame=True)
        data = digits.frame
        data.rename(columns={"target": "Label"}, inplace=True)

    st.write("🔍 Data Sample:")
    st.dataframe(data.head())

    # Split dataset for training and testing
    X = data.drop(columns=["Label"])
    y = data["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model selection for the demo
    model_dict = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Multinomial Naive Bayes": MultinomialNB(),
        "AdaBoost Classifier": AdaBoostClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
        "Support Vector Classification": SVC(),
        "Multi-layer Perceptron": MLPClassifier(),
        "Extra Trees Classifier": ExtraTreesClassifier()
    }

# Fit the selected model (based on user selection)
    model = model_dict.get(algorithm)
    model.fit(X_train, y_train)

    # Performance Metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("📝 Performance Metrics:")
    st.write(f"**Accuracy**: {accuracy:.2f}")
    st.write("**Classification Report**:")
    st.text(classification_report(y_test, y_pred))

# Confusion Matrix
    st.subheader("📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)

# Learning Curve Plot
    st.subheader("📈 Learning Curve:")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label="Training accuracy", color="blue")
    plt.plot(train_sizes, test_mean, label="Cross-validation score", color="green")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    st.pyplot(plt)
if __name__ == "__main__":
    run()


