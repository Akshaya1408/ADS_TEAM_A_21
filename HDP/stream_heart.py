import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load the trained models
models = {
    #   'K-Nearest Neighbors (KNN)': joblib.load('knn_model.pkl'),
    'Random Forest': joblib.load('random_forest_model.pkl'),
    # 'Decision Tree': joblib.load('decision_tree_model.pkl'),
    # 'Logistic Regression': joblib.load('logistic_regression.pkl')

    
}

# Load the dataset with known labels (ground truth)
# Replace 'your_dataset.csv' with your actual dataset file containing labels
# Assuming the dataset has the same columns as used in model training
# For this example, let's assume the dataset has columns: 'age', 'sex', 'cp', 'trestbps', 'chol', 'target'
dataset = pd.read_csv('heart_cleveland_upload.csv')

def predict_heart_disease(model, age, sex, cp, trestbps, chol):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol'])
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title('Heart Disease Prediction')

    model_choice = st.selectbox("Select Model", list(models.keys()))

    age = st.slider("Age", 1, 100, 25)
    sex = st.radio("Sex", ('Male', 'Female'))
    cp = st.selectbox("Chest Pain Type", (1, 2, 3, 4))  # You might need to adjust this
    trestbps = st.slider("Resting Blood Pressure", 50, 250, 120)
    chol = st.slider("Cholesterol Level", 100, 700, 200)

    sex_encoded = 1 if sex == 'Male' else 0

    if st.button("Predict"):
        selected_model = models[model_choice]
        prediction = predict_heart_disease(selected_model, age, sex_encoded, cp, trestbps, chol)
        
        # Calculate accuracy using the selected model on the dataset
        true_labels = dataset['target']  # Replace 'target' with your actual label column name
        predicted_labels = selected_model.predict(dataset.drop(columns=['target']))
        accuracy = accuracy_score(true_labels, predicted_labels)

        if prediction[0] == 1:
            st.warning("Heart Disease Present")
        else:
            st.success("No Heart Disease")
        
        st.write(f"Accuracy of {model_choice}: {accuracy:.2f}")

if __name__ == '__main__':
    main()
