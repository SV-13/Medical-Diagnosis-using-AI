import streamlit as st
import pickle
import os
import joblib
from streamlit_option_menu import option_menu

# Change Name & Logo
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️")

# Hiding Streamlit add-ons
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Define model directory
MODEL_DIR = r'C:\Users\sujal\OneDrive\Documents\Projects\Medical diagnosis using AI\Data\models'

# Function to load the selected model
def load_model(disease, model_type):
    model_filename = f"{disease}_data.csv_{model_type}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model file: {str(e)}")
        return None


# Disease and model selection
st.sidebar.title("Disease Prediction")
diseases = ['Diabetes', 'Heart Disease']
models = ['Logistic Regression', 'Random Forest', 'SVM']

selected_disease = st.sidebar.selectbox('Select a Disease', diseases)
selected_model_type = st.sidebar.selectbox('Select Model Type', models)

# Load the selected model
model = load_model(selected_disease.lower().replace(" ", "_"), selected_model_type.lower().replace(" ", "_"))

# Function for displaying input fields
def display_input_fields(inputs):
    return [st.number_input(label, key=key, help=tooltip, step=1) for label, tooltip, key in inputs]

# Function to predict disease
# Function to predict disease with friendly messages
def predict_disease(model, features, disease_name):
    if model:
        prediction = model.predict([features])
        if prediction[0] == 1:
            st.error(f"🚨 The person **has {disease_name}**.")
            if disease_name == "diabetes":
                st.info("💡 *Tip:* Maintain a balanced diet, exercise regularly, and monitor blood sugar levels.")
            elif disease_name == "heart disease":
                st.info("💡 *Tip:* Avoid smoking, reduce cholesterol, manage stress, and stay active.")
        else:
            st.success(f"✅ The person **does not have {disease_name}**.")
            st.balloons()
            st.info("🎉 *Tip:* Keep up the healthy lifestyle! Regular checkups are still important.")
    else:
        st.error("Model is not loaded properly.")

# UI for each disease prediction
if selected_disease == 'Diabetes':
    st.title('Diabetes Prediction')
    features = display_input_fields([
        ('Number of Pregnancies', 'Enter number of times pregnant', 'Pregnancies'),
        ('Glucose Level', 'Enter glucose level', 'Glucose'),
        ('Blood Pressure value', 'Enter blood pressure value', 'BloodPressure'),
        ('Skin Thickness value', 'Enter skin thickness value', 'SkinThickness'),
        ('Insulin Level', 'Enter insulin level', 'Insulin'),
        ('BMI value', 'Enter Body Mass Index value', 'BMI'),
        ('Diabetes Pedigree Function value', 'Enter diabetes pedigree function value', 'DiabetesPedigreeFunction'),
        ('Age of the Person', 'Enter age of the person', 'Age')
    ])
    if st.button('Predict Diabetes'):
        predict_disease(model, features, "diabetes")

elif selected_disease == 'Heart Disease':
    st.title('Heart Disease Prediction')
    features = display_input_fields([
        ('Age', 'Enter age of the person', 'age'),
        ('Sex (1 = male; 0 = female)', 'Enter sex of the person', 'sex'),
        ('Chest Pain types (0, 1, 2, 3)', 'Enter chest pain type', 'cp'),
        ('Resting Blood Pressure', 'Enter resting blood pressure', 'trestbps'),
        ('Serum Cholesterol in mg/dl', 'Enter serum cholesterol', 'chol'),
        ('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', 'Enter fasting blood sugar', 'fbs'),
        ('Resting ECG results (0, 1, 2)', 'Enter resting ECG results', 'restecg'),
        ('Maximum Heart Rate achieved', 'Enter maximum heart rate', 'thalach'),
        ('Exercise Induced Angina (1 = yes; 0 = no)', 'Enter exercise induced angina', 'exang'),
        ('ST depression induced by exercise', 'Enter ST depression value', 'oldpeak'),
        ('Slope of the peak exercise ST segment (0, 1, 2)', 'Enter slope value', 'slope'),
        ('Major vessels colored by fluoroscopy (0-3)', 'Enter number of major vessels', 'ca'),
        ('Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)', 'Enter thal value', 'thal')
    ])
    if st.button('Predict Heart Disease'):
        predict_disease(model, features, "heart disease")
