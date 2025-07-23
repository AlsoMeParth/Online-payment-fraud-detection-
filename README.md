# 🌟 Online Payment Fraud Detection

A robust fraud detection pipeline leveraging **Random Forest** with SMOTE resampling and optimized threshold tuning for real-time payment fraud detection.

---

## 🚀 Project Overview

This project demonstrates a complete end-to-end workflow for detecting online payment fraud, broken into modular stages using Jupyter notebooks and Python scripts:

1. `01_eda.ipynb` - Exploratory Data Analysis (EDA) and cleaning of the dataset. This notebook uses the original dataset `new_file.csv`.
2. `02_modeling.ipynb` - Imports the cleaned data and performs preprocessing, model selection, threshold tuning, and evaluation between Random Forest and Logistic Regression.
3. `main.py` - End-to-end pipeline automation: preprocessing, training, saving model
4. `app.py` - Streamlit app for real-time fraud prediction through a simple web interface

---

## 🔍 Key Features

* Data cleaning, encoding, and scaling
* SMOTE for class imbalance handling
* Random Forest and Logistic Regression comparison
* Threshold tuning (0.7) for optimizing precision vs. recall
* Precision-Recall curve visualization
* Pickled model and pipeline for reuse

---

## 📂 Repository Structure

```
.
├── .devcontainer         # (Optional VSCode environment setup)
├── .gitattributes        # Git settings
├── .gitignore            # Ignored files
├── 01_eda.ipynb          # Exploratory data analysis and data cleaning
├── 02_modeling.ipynb     # Model training and evaluation using cleaned data
├── readme.md             # Project documentation (this file)
├── app.py                # Streamlit app for inference
├── main.py               # Full pipeline (preprocessing + model training)
├── model.pkl             # Trained Random Forest model
├── pipeline.pkl          # Preprocessing pipeline
├── requirements.txt      # Dependencies
```

---

## 💡 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/AlsoMeParth/Online-payment-fraud-detection-.git
cd Online-payment-fraud-detection-
```

### 2. Install required libraries

```bash
pip install -r requirements.txt
```

### 3. Run Notebooks (for analysis & modeling)

Use Jupyter Notebook or JupyterLab to open:

* `01_eda.ipynb` to perform EDA and clean the dataset `new_file.csv`
* `02_modeling.ipynb` to train and evaluate models using the cleaned data

### 4. Run the Pipeline Script

To retrain or reproduce the model:

```bash
python main.py
```

This will output `model.pkl` and `pipeline.pkl`.

### 5. Launch the Streamlit App

```bash
streamlit run app.py
```

This opens an interactive web UI for uploading input files and visualizing fraud predictions.

---

## 📁 Download Dataset

Due to GitHub's 100MB file limit, large datasets are hosted externally.

You can download the required dataset file `new_file.csv` from the following Google Drive link:

👉 [Download from Google Drive](https://drive.google.com/file/d/1rswZWQeAW0LtIbPaKJ97S9dUFBh42wj6/view?usp=drive_link)

---

## 📊 Model Performance

| Model               | Average Precision (AP) |
| ------------------- | ---------------------- |
| Random Forest       | **1.00**               |
| Logistic Regression | 0.55                   |

* **Optimal Threshold**: 0.7
* High recall ensures nearly all frauds are detected
* Precision improved significantly by threshold tuning

---

## 🔄 Next Steps

* Monitor the deployed model in real-world scenarios
* Enhance feature set with behavior profiling, device IDs, etc.
* Explore advanced algorithms (e.g., XGBoost, LSTM)
* Package the app with Docker or scale deployment for larger data streams

---

## 📄 Acknowledgements

* SMOTE for balancing classes
* Scikit-learn for modeling pipeline
* Streamlit for interactive deployment
* Jupyter Notebooks for iterative analysis

---

## 🔗 Connect

Issues and contributions welcome! Open a PR or issue if you have suggestions or questions.

Thank you for checking out this project! 🚀
