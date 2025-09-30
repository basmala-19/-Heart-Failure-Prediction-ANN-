# Heart Failure Prediction using ANN

This project uses an **Artificial Neural Network (ANN)** to predict the likelihood of heart disease based on patient data.  
The dataset comes from [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

---

## 📌 Project Structure
- `heart-failure-prediction-ann.ipynb` → Jupyter Notebook with model training and evaluation  
- `app.py` → Streamlit app for user-friendly predictions  
- `requirements.txt` → Dependencies  

---

## ⚙️ Features
- Data preprocessing (categorical encoding, normalization)
- Artificial Neural Network (ANN) for classification
- Model evaluation (Accuracy, F1-score, Classification Report)
- Streamlit web app for real-time predictions

---

## 🚀 How to Run
### 1. Clone the repository
```bash
git clone https://github.com/yourusername/heart-failure-prediction-ann.git
cd heart-failure-prediction-ann
```
### 2. Install dependencies
```
pip install -r requirements.txt
```
### 3. Run Streamlit app
```
streamlit run app.py

```
## 📊 Model Performance

```

- Accuracy: 88%

- F1-score: 89%

```
## 🖥️ Example (Streamlit UI)
```
- Input patient data (Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, etc.)

- Get instant prediction if the patient is likely to have heart disease.
```