# 🧠 Medical Diagnosis using AI

An AI-powered web application that predicts the likelihood of **Diabetes** and **Heart Disease** using machine learning models.

🔗 **Live App:**  
[https://medical-diagnosis-using-ai-cac8wkoch5henkawsauaow.streamlit.app/](https://medical-diagnosis-using-ai-cac8wkoch5henkawsauaow.streamlit.app/)

---

## ✨ Features

- Predict **Diabetes** and **Heart Disease**
- Choose between 3 model types: `Logistic Regression`, `Random Forest`, `SVM`
- Built with **Streamlit** for an interactive UI
- Backend powered by **Scikit-learn** models
- Input fields with tooltips for ease of use
- Balloons and success tips on predictions

---

## 🧰 Tech Stack

| Layer         | Technology             |
|---------------|-------------------------|
| Frontend      | Streamlit               |
| Backend       | Python                  |
| ML Models     | Scikit-learn            |
| Deployment    | Streamlit Cloud         |
| File Handling | Pickle (.pkl models)    |

---

## 🗂️ Project Structure

```
Medical Diagnosis using AI/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── Data/
│   └── models/
│       ├── diabetes_data.csv_logistic_regression.pkl
│       ├── heart_disease_data.csv_random_forest.pkl
│       └── ...
```

---

## 💻 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/SV-13/Medical-Diagnosis-using-AI.git
   cd Medical-Diagnosis-using-AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 📦 Deployment

The application is deployed using **Streamlit Cloud**. You can access it directly via:

🔗 [medical-diagnosis-using-ai.streamlit.app](https://medical-diagnosis-using-ai-cac8wkoch5henkawsauaow.streamlit.app/)

---

## 🤖 Supported Models

### Diabetes
- Logistic Regression
- Random Forest
- SVM

### Heart Disease
- Logistic Regression
- Random Forest
- SVM

Each model is saved along with:
- `feature_names.pkl`
- `scaler.pkl`

---

## 👤 Author

**Sujal Verma**  
[GitHub](https://github.com/SV-13) | [LinkedIn](https://www.linkedin.com/in/sujal-verma-816190269/)

---

## 📝 License

This project is licensed under the **MIT License**.

