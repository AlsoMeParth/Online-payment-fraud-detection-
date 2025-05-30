# 🔐 Online Payment Fraud Detection using Machine Learning

This project demonstrates a complete machine learning pipeline to detect fraudulent transactions in an online payment system. It includes data preprocessing, feature engineering, model training and evaluation, and deployment using a Streamlit web app.

---

## ✨ Features

- Clean and preprocess transactional data
- Handle class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**
- Encode categorical features using **One-Hot Encoding**
- Apply **StandardScaler** for feature normalization
- Train and compare **Logistic Regression** and **Random Forest Classifier**
- Evaluate model using **Accuracy, Precision, Recall, F1 Score, Confusion Matrix**
- Deploy a **Streamlit web application** for real-time fraud prediction

---

## 📄 Files in the Repository

| File | Description |
|------|-------------|
| `model.ipynb` | Jupyter Notebook with complete ML workflow |
| `model_bundle.pkl` | Saved model file (Random Forest) using Joblib |
| `app.py` | Streamlit app for real-time fraud prediction |
| `requirements.txt` | Text file containing name of libraries required to run the model locally |
| `README.md` | Project overview and usage guide |

---

## 👨‍💻 Technologies Used

- **Python**: Programming language
- **Pandas, NumPy**: Data manipulation and analysis
- **Matplotlib, Seaborn**: Data visualization
- **scikit-learn**: Machine learning models and metrics
- **imblearn**: For SMOTE oversampling
- **Streamlit**: Web app development
- **Joblib**: Model serialization

---

## 🌐 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fraud-detection-ml.git
cd fraud-detection-ml
```

### 2. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook model.ipynb
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```

---

## 🚀 Future Improvements
- Integrate real-time transaction feed
- Use deep learning models (e.g., LSTM) for sequential fraud patterns
- Add SHAP or LIME for explainable AI
- Include alerts and logging system for fraud triggers

---


