import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, chi2

st.set_page_config(layout="wide")

def display_homepage():
    st.subheader("Name      : Chew Yang ")
    st.subheader("Matric Id : CB20172 ")
    st.subheader("Algorithm")
    st.write("Algorithm that apply in my project is Random Forest and Support Vector Machine")
    st.write("- Random Forest")
    st.write("- Support Vector Machine")
    st.write("")
    
    # Add a link to a website
    st.subheader("Dataset")
    st.write("This is the dataset that I obtained. This dataset is the dataset of cars purchase decision at Kaggle website.")
    st.markdown("https://www.kaggle.com/datasets/gabrielsantello/cars-purchase-decision-dataset")
    
def prediction_page(random_forest_model, svm_model, X):
    st.subheader("Prediction")

    # Model Selection
    model_selection = st.radio("Select Algorithm", options=['Random Forest', 'SVM'])

    # Age
    age = st.slider("Age", min_value=int(X['Age'].min()), max_value=int(X['Age'].max()), step=1)

    # Gender
    gender = st.radio("Gender", options=['Male', 'Female'])

    # Annual Salary
    annual_salary = st.slider("Annual Salary", min_value=int(X['AnnualSalary'].min()), max_value=int(X['AnnualSalary'].max()), step=1)

    # Convert gender to numeric
    gender_numeric = 1 if gender == 'Male' else 0

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({'Age': [age], 'AnnualSalary': [annual_salary], 'Gender': [gender_numeric]})

    # Make the prediction based on the selected model
    prediction = None
    if model_selection == 'Random Forest' and random_forest_model is not None:
        prediction = random_forest_model.predict(input_data)
    elif model_selection == 'SVM' and svm_model is not None:
        prediction = svm_model.predict(input_data)

    # Display the prediction
    st.subheader("Prediction Result")
    if prediction is not None:
        if prediction[0] == 0:
            st.write("Not purchased")
        elif prediction[0] == 1:
            st.write("Purchased")
    else:
        st.write("No model selected or model not trained yet.")

def display_eda_page():
    st.title("Exploratory Data Analysis (EDA)")

    file_path = "car_data.csv"
    try:
        # Read excel file
        data = pd.read_csv(file_path)
        st.subheader("Original Data")
        st.dataframe(data)

        # Drop User ID column
        st.subheader("Data with Column 'User ID' Dropped")
        data_dropped = data.drop('User ID', axis=1)
        st.dataframe(data_dropped)
        
        # Drop function to drop NaN data
        st.subheader("Table of Data with Missing Values Dropped")
        data_dropped_na = data_dropped.dropna()
        st.dataframe(data_dropped_na)

        # Label encode "Gender" column
        st.subheader("Data with Encoded Gender")
        st.subheader("Male=1  Female=0")
        data_encoded = data_dropped_na.copy()
        label_encoder = LabelEncoder()
        data_encoded['Gender'] = label_encoder.fit_transform(data_encoded['Gender'])
        st.dataframe(data_encoded)

        # Store data in session state
        st.session_state.dataG = data_encoded

        # Display scatter plot
        display_scatterplot()

        # Display bar chart plot
        display_bar_chart()

    except FileNotFoundError:
        st.error("File Error!")

import pandas as pd

import pandas as pd

def display_SVM_page():
    st.title("Algorithm")
    st.write("")
    st.subheader("Support Vector Machine")

    # Call the Support Vector Machine function
    data = st.session_state.dataG  # Assuming 'dataG' is the preprocessed data
    X = data[['Age', 'AnnualSalary', 'Gender']]  # Assuming 'Age', 'AnnualSalary', and 'Gender' are the features
    y = data['Purchased']  # Assuming 'Purchased' is the target variable

    # Perform feature selection
    selector = SelectKBest(score_func=chi2, k=2)  # Specify the number of features to select
    X_selected = selector.fit_transform(X, y)

    # Train and evaluate the SVM model on the selected features
    result = support_vector_machine(X_selected, y)

    # Display the results
    st.subheader("Confusion Matrix")
    cmSVM = np.array([[result[0][2], result[0][1]], [result[0][3], result[0][0]]])
    st.write("Confusion Matrix:")
    st.write(cmSVM)

    st.write("True Positive (TP):", result[0][0])
    st.write("False Positive (FP):", result[0][1])
    st.write("True Negative (TN):", result[0][2])
    st.write("False Negative (FN):", result[0][3])
    st.subheader("Precision")
    st.write(result[1])
    st.subheader("Recall")
    st.write(result[2])
    st.subheader("F-Score")
    st.write(result[3])
    st.subheader("Accuracy")
    st.write(result[4])
    
    # Print the results
    print("Confusion Matrix:")
    print(cmSVM)
    print("True Positive (TP):", result[0][0])
    print("False Positive (FP):", result[0][1])
    print("True Negative (TN):", result[0][2])
    print("False Negative (FN):", result[0][3])
    print("Precision:", result[1])
    print("Recall:", result[2])
    print("F-Score:", result[3])
    print("Accuracy:", result[4])

    # Display the selected features
    st.subheader("Important Features")
    selected_feature_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_feature_indices]
    st.write(selected_features)

    # Display feature importance percentages
    st.subheader("Feature Importance")
    importances = selector.scores_
    importance_percentages = importances / np.sum(importances)
    importance_data = pd.DataFrame({'Feature': X.columns, 'Importance': importance_percentages})
    importance_data_formatted = importance_data.round(10)
    importance_data_formatted['Importance'] = importance_data_formatted['Importance'].map("{:.10f}".format)
    st.table(importance_data_formatted.style.set_properties(**{'text-align': 'left'}))
    print(importance_data_formatted)


def display_forest_page():
    st.title("Algorithm")
    st.write("")
    st.subheader("Random Forest")
    st.write("")
    
    # Call the Random Forest function
    data = st.session_state.dataG  # Assuming 'dataG' is the preprocessed data
    X = data[['Age', 'AnnualSalary','Gender']]  # Assuming 'Age' and 'AnnualSalary' are the features
    y = data['Purchased']  # Assuming 'TargetVariable' is the target variable
    result = random_forest_classifier(X, y)

    # Display the results
    st.subheader("Confusion Matrix")

    # Print the confusion matrix
    cm = np.array([[result[0][0], result[0][1]], [result[0][2], result[0][3]]])
    st.write("Confusion Matrix:")
    st.write(cm)

    st.write("True Positive (TP):", result[0][0])
    st.write("False Positive (FP):", result[0][1])
    st.write("True Negative (TN):", result[0][2])
    st.write("False Negative (FN):", result[0][3])
    
    st.subheader("Precision")
    st.write(result[1])
    st.subheader("Recall")
    st.write(result[2])
    st.subheader("F-Score")
    st.write(result[3])
    st.subheader("Accuracy")
    st.write(result[4])
    
    st.subheader("Feature Importance")
    st.write(result[5])

def display_scatterplot():
    st.subheader("Scatter Plot")
    data = st.session_state.dataG
    plt.scatter(data['Age'], data['AnnualSalary'])
    plt.xlabel('Age')
    plt.ylabel('AnnualSalary')
    st.pyplot()

def display_bar_chart():
    st.subheader("Bar Chart")
    data = st.session_state.dataG

    male_data = data[data['Gender'] == 1]  # Filter data for male gender
    female_data = data[data['Gender'] == 0]  # Filter data for female gender

    fig, ax = plt.subplots()
    ax.bar(male_data['Age'], male_data['AnnualSalary'], color='blue', alpha=0.7, label='Male')
    ax.bar(female_data['Age'], female_data['AnnualSalary'], color='pink', alpha=0.7, label='Female')
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Salary")
    ax.legend()
    st.pyplot(fig)

def random_forest_classifier(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize a Random Forest classifier
    classifier = RandomForestClassifier()

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate precision, recall, f-score, and accuracy
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate feature importances
    importance_scores = classifier.feature_importances_

    # Create a DataFrame to store feature importances
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importance_scores})

    # Sort the features by importance in descending order
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    # Return the results
    return (tn, fp, fn, tp), precision, recall, f_score, accuracy, feature_importances

def train_random_forest_model(X, y):
    # Create a Random Forest classifier
    random_forest_classifier = RandomForestClassifier()

    # Train the model
    random_forest_classifier.fit(X, y)

    return random_forest_classifier

def support_vector_machine(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize an SVM classifier
    classifier = SVC()

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate precision, recall, f-score, and accuracy
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Return the results
    return (tn, fp, fn, tp), precision, recall, f_score, accuracy

def train_svm_model(X, y):
    # Create an SVM classifier with the sigmoid kernel
    svm_classifier = svm.SVC(kernel='sigmoid', probability=True)  # Set probability=True to enable predict_proba()

    # Train the model
    svm_classifier.fit(X, y)

    return svm_classifier

def main():
    # Set the option to disable the warning
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Sidebar navigation
    page_options = ["Home", "EDA", "RANDOM FOREST","SUPPORT VECTOR MACHINE","PREDICTION"]
    selected_page = st.sidebar.selectbox("Select a page", page_options)

    # Train the Random Forest model
    if 'dataG' in st.session_state:
        data = st.session_state.dataG
        X = data[['Age', 'AnnualSalary','Gender']]
        y = data['Purchased']
        random_forest_model = train_random_forest_model(X,y)
        svm_model = train_svm_model(X, y)
    else:
        random_forest_model = None

    # Display selected page
    if selected_page == "Home":
        display_homepage()
    elif selected_page == "EDA":
        display_eda_page()
    elif selected_page == "RANDOM FOREST":
        display_forest_page()
    elif selected_page == "SUPPORT VECTOR MACHINE":
        display_SVM_page()
    elif selected_page == "PREDICTION":
        if random_forest_model is not None or svm_model is not None:
            prediction_page(random_forest_model, svm_model, X)
        else:
            st.write("Random Forest model not trained yet. Please go to 'RANDOM FOREST' page and train the model first.")

if __name__ == '__main__':
    main()
