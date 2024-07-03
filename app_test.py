
# from distutils.command import clean
import streamlit as st
import pandas as pd
import numpy as np
# import numpy as np
import matplotlib.pyplot as plt
import joblib
# from sklearn.preprocessing import LabelEncoder
import cleaning
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer





# Load the model and vectorizer for initial distinguish of prediction
svm_model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load the models and the input columns
disease_model = joblib.load('disease_prediction_model.pkl')
disease_columns = joblib.load('disease_prediction_model_columns.pkl')


# Set page configuration to wide mode
st.set_page_config(
    layout="wide"  # Set the layout to wide mode
)

# Streamlit app
st.title('Windy City Health Coders - Medical Transcripts GenAI')

# Add custom CSS for font sizes and input box styling
st.markdown("""
    <style>
        .big-font {
            font-size: 24px !important;
            color: #333;
        }
        .medium-font {
            font-size: 18px !important;
            color: #666;
        }
        .text-input-label {
            font-size: 24px !important;
            color: #333;
        }
        .dataframe {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: #f9f9f9;
        }
        .text-input {
            font-size: 18px !important;
            padding: 10px;
            border-radius: 5px;
            border: 2px solid #007bff;
            background-color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# File uploader widget with larger font
st.markdown('<p class="big-font">Upload a CSV file</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type="csv")

# Initialize the user query as None
user_query = None

if uploaded_file is not None:
    # Uploaded CSV cleaning
    df_try = pd.read_csv(uploaded_file)
    cleaned_data = pd.read_csv("df_final.csv")
    
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_colwidth', None)  # Show full length of each column
    
    # # Display the ORIGINAL dataset with a larger font size
    # st.markdown('<p class="big-font">Here is the UPLOADED dataset:</p>', unsafe_allow_html=True)
    # st.dataframe(df_try, use_container_width=True)  # Use st.dataframe() with full container width

    # # Display the EXTRACTED dataset with a larger font size
    # st.markdown('<p class="big-font">Here is the EXTRACTED dataset:</p>', unsafe_allow_html=True)
    # st.dataframe(cleaned_data, use_container_width=True)  # Use st.dataframe() with full container width
    
    # Add a text input box for the user query with a larger font size and placeholder text
    st.markdown('<p class="text-input-label">Ask me anything about this dataset: (if you need disease prediction, type "please predict".)</p>', unsafe_allow_html=True)
    user_query = st.text_input("AAA", placeholder="Type your question and press Enter...", label_visibility="hidden", help="Enter your question here", key="user_query")

else:
    st.write("No files uploaded.")

# Check if the user_query is not empty
if user_query:
    
    if uploaded_file is not None:
        
        new_query = [user_query]
        X_new = vectorizer.transform(new_query)
        prediction = svm_model.predict(X_new)

        if prediction == 1 and 'trend' not in user_query.lower(): # ML prediction part
            st.write(f"The query '{user_query}' contains a prediction-related keyword.")
            
            if "diabetes" in user_query.lower(): # diabetes model
                st.write("Yes, we can. Here is the result:")
                show_df = cleaned_data[["full_name", "diabetes_prediction","diabetes_probability"]]
                st.dataframe(show_df, use_container_width=False)  # Use st.dataframe() with full container width

                st.markdown('<p class="text-input-label">Which patient are you specifically looking for?</p>', unsafe_allow_html=True)
                temp_user_query = st.text_input("AAA", placeholder="Please type his/her full name.", label_visibility="hidden", help="Enter your question here", key="temp_user_query")
                
                # Check if the user query is in the 'full_name' column of the DataFrame
                if temp_user_query:
                    if temp_user_query in show_df["full_name"].values:
                        # Get the row with the specific patient's details
                        patient_row = show_df[show_df["full_name"] == temp_user_query]
                        st.write(f"Here are the prediction and probability for {temp_user_query}:")
                        st.dataframe(patient_row, use_container_width=False)
                    else:
                        st.write(f"No patient found with the name '{temp_user_query}'. Please check the name and try again.")
                else:
                    st.write("Please input the full name of the patient.")


            else: # disease prediction model
                st.write("This is the disease prediction model.")
                st.markdown('<p class="text-input-label">Please enter your symptoms and medical history here.</p>', unsafe_allow_html=True)
                temp_user_query = st.text_input("AAA", placeholder="Please enter your symptoms and medical history here...", label_visibility="hidden", help="Enter your question here", key="temp_user_query")

                if temp_user_query:
                    new_row = {
                        'Unnamed: 0': 4999,
                        'description': temp_user_query,
                        'medical_specialty': "",
                        'sample_name': "",
                        'transcription': "",
                        'keywords': "",
                        'first_name': "",
                        'last_name': ""
                        }
                    
                    df_try.loc[len(df_try)] = new_row
                    
                    # Start Cleaning the Data
                    df_try.dropna(inplace=True)

                    # Apply the extraction function to each relevant column
                    df_try['age_description'] = df_try['description'].apply(cleaning.extract_age)
                    df_try['age_transcription'] = df_try['transcription'].apply(cleaning.extract_age)

                    # Fill the missing values from both columns
                    df_try['age'] = df_try[['age_description', 'age_transcription']].bfill(axis=1).iloc[:, 0]

                    # Drop the intermediate columns
                    df_try.drop(columns=['age_description', 'age_transcription'], inplace=True)

                    df_try['diabetes'] = df_try.apply(
                        lambda row: cleaning.has_diabetes(row['keywords']) or cleaning.has_diabetes(row['description']) or cleaning.has_diabetes(row['transcription']),
                        axis=1
                    ).astype(int)

                    df_try['hypertension'] = df_try.apply(
                        lambda row: cleaning.has_hypertension(row['keywords']) or cleaning.has_hypertension(row['description']) or cleaning.has_hypertension(row['transcription']),
                        axis=1
                    ).astype(int)

                    df_try['heart_disease_type'] = df_try.apply(
                        lambda row: cleaning.extract_heart_disease_type(row['keywords']) or cleaning.extract_heart_disease_type(row['description']) or cleaning.extract_heart_disease_type(row['transcription']),
                        axis=1
                    )

                    df_try['has_heart_disease'] = df_try['heart_disease_type'].notnull().astype(int)

                    df_try['heart_disease_year'] = df_try.apply(
                        lambda row: cleaning.extract_year(row['transcription']) if row['has_heart_disease'] else None,
                        axis=1
                    )
                    df_try['heart_disease_type'].fillna('Other', inplace=True)

                    df_try['gender_first_name'] = df_try['first_name'].apply(cleaning.extract_gender_from_name)

                    df_try['gender_description'] = df_try['description'].apply(cleaning.extract_gender)
                    df_try['gender_transcription'] = df_try['transcription'].apply(cleaning.extract_gender)
                    df_try['gender_keywords'] = df_try['keywords'].apply(cleaning.extract_gender)
                    df_try['gender_medical_specialty'] = df_try['medical_specialty'].apply(cleaning.extract_gender)
                    df_try['gender_sample_name'] = df_try['sample_name'].apply(cleaning.extract_gender)

                    df_try['gender'] = df_try[
                        ['gender_first_name', 'gender_description', 'gender_transcription', 'gender_keywords', 'gender_medical_specialty', 'gender_sample_name']
                    ].apply(cleaning.combine_genders, axis=1)

                    df_try.drop(columns=['gender_first_name', 'gender_description', 'gender_transcription', 'gender_keywords', 'gender_medical_specialty', 'gender_sample_name'], inplace=True)

                    df_try['visit_type'] = df_try['description'].apply(cleaning.extract_visit_type)
                    df_try['procedure_type'] = df_try['description'].apply(cleaning.extract_procedure_type)
                    df_try['admission_type'] = df_try['transcription'].apply(cleaning.extract_admission_type)

                    df_try['body_parts_description'] = df_try['description'].apply(cleaning.extract_body_parts)
                    df_try['body_parts_transcription'] = df_try['transcription'].apply(cleaning.extract_body_parts)
                    df_try['body_parts_keywords'] = df_try['keywords'].apply(cleaning.extract_body_parts)

                    df_try['body_parts'] = df_try[['body_parts_description', 'body_parts_transcription', 'body_parts_keywords']].apply(
                        lambda x: ', '.join(filter(None, set(x.dropna()))), axis=1
                    )
                    df_try['body_parts'] = df_try['body_parts'].apply(lambda x: ', '.join(sorted(set(x.split(', ')))) if pd.notnull(x) else None)

                    df_try.drop(columns=['body_parts_description', 'body_parts_transcription', 'body_parts_keywords'], inplace=True)
                    df_try['body_parts'].replace('', 'Other', inplace=True)

                    df_try['symptoms_description'] = df_try['description'].apply(cleaning.extract_symptoms)
                    df_try['symptoms_transcription'] = df_try['transcription'].apply(cleaning.extract_symptoms)
                    df_try['symptoms_keywords'] = df_try['keywords'].apply(cleaning.extract_symptoms)
                    df_try['symptoms_medical_specialty'] = df_try['medical_specialty'].apply(cleaning.extract_symptoms)

                    # Combine the extracted symptoms into a single column
                    df_try['symptoms'] = df_try[
                        [
                            'symptoms_description', 'symptoms_transcription', 'symptoms_keywords', 'symptoms_medical_specialty'
                        ]
                    ].apply(lambda x: ', '.join(filter(None, set(x.dropna()))), axis=1)

                    # Sort and deduplicate the combined symptoms
                    df_try['symptoms'] = df_try['symptoms'].apply(lambda x: ', '.join(sorted(set(x.split(', ')))) if pd.notnull(x) else 'None')

                    # Drop the intermediate columns
                    df_try.drop(columns=[
                        'symptoms_description', 'symptoms_transcription', 'symptoms_keywords', 'symptoms_medical_specialty'
                    ], inplace=True)

                    df_try['symptoms'].replace('', 'Other', inplace=True)

                    st.write("The following process would take around 10 mins...")

                    # Apply the extraction function to each relevant column
                    df_try['diagnosis_description'] = df_try['description'].apply(cleaning.extract_disease)
                    df_try['diagnosis_transcription'] = df_try['transcription'].apply(cleaning.extract_disease)
                    df_try['diagnosis_keywords'] = df_try['keywords'].apply(cleaning.extract_disease)
                    df_try['diagnosis_sample_name'] = df_try['sample_name'].apply(cleaning.extract_disease)
                    df_try['diagnosis_medical_specialty'] = df_try['medical_specialty'].apply(cleaning.extract_disease)
                    df_try['diagnosis_symptoms'] = df_try['symptoms'].apply(cleaning.extract_disease)
                    df_try['diagnosis_procedure_type'] = df_try['procedure_type'].apply(cleaning.extract_disease)
                    df_try['diagnosis_body_parts'] = df_try['body_parts'].apply(cleaning.extract_disease)
                    df_try['diagnosis_visit_type'] = df_try['visit_type'].apply(cleaning.extract_disease)

                    # Combine the extracted diagnoses into a single column
                    df_try['diagnosis'] = df_try[
                        [
                            'diagnosis_description', 'diagnosis_transcription', 'diagnosis_keywords', 
                            'diagnosis_sample_name', 'diagnosis_medical_specialty', 'diagnosis_symptoms',
                            'diagnosis_procedure_type', 'diagnosis_body_parts', 'diagnosis_visit_type'
                        ]
                    ].apply(lambda x: ', '.join(filter(None, set(x.dropna()))), axis=1)

                    # Sort and deduplicate the combined diagnoses
                    df_try['diagnosis'] = df_try['diagnosis'].apply(lambda x: ', '.join(sorted(set(x.split(', ')))) if pd.notnull(x) else 'None')

                    # Drop the intermediate columns
                    df_try.drop(columns=[
                        'diagnosis_description', 'diagnosis_transcription', 'diagnosis_keywords', 
                        'diagnosis_sample_name', 'diagnosis_medical_specialty', 'diagnosis_symptoms',
                        'diagnosis_procedure_type', 'diagnosis_body_parts', 'diagnosis_visit_type'
                    ], inplace=True)
                    df_try['diagnosis'].replace('','Other', inplace=True)

                    df_try['heart_disease_type'].fillna('Other', inplace=True)
                    df_try['diagnosis'].fillna('Other', inplace=True)
                    df_try['symptoms'].fillna('Other', inplace=True)
                    df_try['gender'].fillna('Other', inplace=True)

                    df_try['age'].fillna(df_try['age'].mean(), inplace=True)
                
                    df_model = df_try[['keywords',  'age', 'diabetes', 'hypertension',
                                    'heart_disease_type', 'has_heart_disease',
                                    'gender', 'visit_type', 'procedure_type',
                                    'admission_type', 'body_parts', 'symptoms', 'diagnosis']]
                    
                    X_other = df_model[['age', 'diabetes', 'hypertension', 'heart_disease_type', 'has_heart_disease', 'gender', 'visit_type', 'procedure_type', 'admission_type', 'body_parts', 'symptoms']]
                    categorical_cols = ['heart_disease_type', 'gender', 'visit_type', 'procedure_type', 'admission_type', 'body_parts', 'symptoms']

                    onehot_encoder = OneHotEncoder(sparse_output=False)
                    X_encoded = onehot_encoder.fit_transform(X_other[categorical_cols])

                    X_numeric = X_other.drop(columns=categorical_cols).values
                    X = np.hstack([X_numeric, X_encoded])

                    mlb = MultiLabelBinarizer()
                    y = mlb.fit_transform(df_model['diagnosis'].str.split(', '))

                    indices = df_model.index.values
                    rfr = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))
                    rfr.fit(X, y)
                    y_pred = rfr.predict(X)
                    # y_pred_proba = disease_model.predict_proba(X)

                    # Get the predicted diagnoses from the multi-label binarizer
                    predicted_diagnosis = ['; '.join(diags) for diags in mlb.inverse_transform(y_pred)]
                    
                    # Add the predicted diagnoses to the original DataFrame
                    df_model['predicted_diagnosis'] = predicted_diagnosis
                    target_value = df_model.iloc[-1]['predicted_diagnosis']

                    st.write(f"Your predicted diseases are: {target_value}")

        else: # EDA related questions (SQL-like queries)
            st.write(f"The query '{user_query}' does not contain any prediction-related keywords.")
            # st.dataframe(cleaned_data, use_container_width=True)  # Use st.dataframe() with full container width
            
            if "most common" in user_query.lower():
                most_common_condition = cleaned_data['diagnosis'].value_counts().idxmax()
                most_common_condition_count = cleaned_data['diagnosis'].value_counts().max()
                st.write(f"The most common medical condition is '{most_common_condition}', with {most_common_condition_count} occurrences.")

            else:
                # Load a zero-shot classification pipeline with DistilBERT model from Hugging Face
                classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
                
                # Define the context
                context = f"This is the question: '{user_query}'"

                # Define candidate labels
                candidate_labels = ["sum/how many", "average", "trend"]

                # Perform zero-shot classification
                result = classifier(context, candidate_labels)
                top_label = result['labels'][0]

                # Print the result
                st.write("The classification for this question is:", top_label)
                st.dataframe(cleaned_data.head(5), use_container_width=True)  # Use st.dataframe() with full container width

                if top_label == "sum/how many":
                    for column in cleaned_data.columns:
                        if column.lower() in user_query.lower():
                            column_sum = cleaned_data[column].sum()
                            st.write(f"The sum of the '{column}' column is {column_sum}, that means a total of {column_sum} patient(s) has(have) been diagnosed with {column} in the dataset and is(are) likely prescribed medication.")

                elif top_label == "average":
                    for column in cleaned_data.columns:
                        if column.lower() in user_query.lower():
                            column_average = round(cleaned_data[column].mean(), 4)
                            st.write(f"The average of the '{column}' column is {column_average}, that means the average {column} of the patients in the dataset is {column_average}.")

                elif top_label == "trend":
                    heart_disease_trend = cleaned_data[cleaned_data['has_heart_disease'] == 1].groupby('heart_disease_year').size()

                    st.write("Trend analysis on heart disease incidence over the years:")
                    st.write(heart_disease_trend)
                    
                    # Choose which one is better!

                    plt.plot(heart_disease_trend)
                    plt.title('Trend Analysis on Heart Disease Incidence Over the Years')
                    plt.xlabel('Year')
                    plt.ylabel('Number of Heart Disease Cases')
                    plt.grid(True)
                    st.pyplot(plt)

                    st.line_chart(heart_disease_trend,
                                width=1000,
                                height=400,
                                use_container_width=False)

                else:
                    st.write("This kind of question not yet supported!")