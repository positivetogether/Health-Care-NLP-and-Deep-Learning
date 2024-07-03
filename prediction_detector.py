import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

import joblib

# Define example sentences for each class
prediction_queries = [
    "What is the likelihood of developing diabetes given these symptoms?",
    "Can you predict the chances of getting hypertension from this data?",
    "Based on my symptoms, what disease might I have?",
    "How likely am I to develop heart disease?",
    "Could you predict the probability of a cancer diagnosis with these symptoms?",
    "What is the risk of having a stroke given my medical history?",
    "Can you forecast my chances of developing asthma?",
    "What are the chances of getting arthritis with my current symptoms?",
    "Can you predict if I will have high cholesterol?",
    "How likely is it for me to develop osteoporosis?",
    "What is the likelihood of contracting pneumonia based on these symptoms?",
    "Can you give me a prediction for my risk of a thyroid disorder?",
    "How likely am I to develop a liver disease?",
    "Can you predict if I will get a kidney disease?",
    "What is the probability of having anemia given my symptoms?",
    "Based on this data, can you predict my chances of having a gallbladder issue?",
    "How likely is it that I have a digestive disorder?",
    "Can you forecast the probability of a skin condition?",
    "What are the chances of developing a neurological disorder from these symptoms?",
    "Can you predict my risk of getting an autoimmune disease?",
    "How likely am I to have a genetic disorder?",
    "Based on my medical history, can you predict the onset of a mental health condition?",
    "What is the likelihood of developing a metabolic syndrome?",
    "Can you forecast the chances of an endocrine disorder?",
    "How likely is it that I will get a respiratory infection?",
    "What is the probability of having a cardiovascular disease?",
    "Can you predict if I will develop a gastrointestinal disorder?",
    "How likely am I to get an infectious disease from these symptoms?",
    "Based on this data, what is my risk of developing an eye condition?",
    "Can you predict the likelihood of an ear infection?",
    "What is the probability of having a musculoskeletal disorder?",
    "How likely am I to develop a blood disorder?",
    "Can you forecast my chances of getting a reproductive system disease?",
    "What are the chances of having a urinary tract infection based on these symptoms?",
    "Can you predict if I will get a dental problem?",
    "How likely is it that I have an immune deficiency?",
    "What is the risk of developing a congenital disorder?",
    "Based on my symptoms, can you predict if I have a parasitic infection?",
    "How likely am I to have an allergic reaction?",
    "Can you forecast the probability of a viral infection?",
    "What are the chances of getting a bacterial infection from these symptoms?",
    "Can you predict my risk of having a fungal infection?",
    "How likely is it that I will develop a chronic disease?",
    "What is the probability of having an acute illness given my symptoms?",
    "Based on this data, can you predict if I have a psychiatric disorder?",
    "How likely am I to get a sexually transmitted infection?",
    "Can you forecast the chances of a digestive tract infection?",
    "What is the risk of developing a sleep disorder?",
    "Can you predict if I will have a circulatory system issue?",
    "How likely is it that I will get a disease from these symptoms?"
]

non_prediction_queries = [
    "What are the common symptoms of diabetes?",
    "Can you provide an analysis of my blood test results?",
    "What is the average age for developing hypertension?",
    "How does one manage heart disease?",
    "What are the stages of cancer development?",
    "Can you give me a summary of my medical history?",
    "What are the common treatments for asthma?",
    "How is arthritis diagnosed?",
    "What are the dietary recommendations for high cholesterol?",
    "How does osteoporosis affect the body?",
    "What are the symptoms of pneumonia?",
    "Can you provide statistics on thyroid disorders?",
    "What are the risk factors for liver disease?",
    "How is kidney disease treated?",
    "What are the causes of anemia?",
    "Can you explain gallbladder function?",
    "What are the symptoms of a digestive disorder?",
    "How is a skin condition diagnosed?",
    "What are the common neurological disorders?",
    "Can you explain autoimmune diseases?",
    "What are genetic disorders?",
    "How is mental health assessed?",
    "What is metabolic syndrome?",
    "What are endocrine disorders?",
    "How do respiratory infections spread?",
    "Can you provide an overview of cardiovascular diseases?",
    "What are the common gastrointestinal disorders?",
    "How are infectious diseases treated?",
    "What are the symptoms of an eye condition?",
    "How is an ear infection diagnosed?",
    "What are musculoskeletal disorders?",
    "What are the symptoms of a blood disorder?",
    "How is a reproductive system disease treated?",
    "What are the common causes of urinary tract infections?",
    "How does one maintain dental health?",
    "What are immune deficiencies?",
    "How are congenital disorders diagnosed?",
    "What are parasitic infections?",
    "How does an allergic reaction occur?",
    "What are the common viral infections?",
    "How are bacterial infections treated?",
    "What are fungal infections?",
    "How are chronic diseases managed?",
    "What is an acute illness?",
    "What are the symptoms of psychiatric disorders?",
    "How are sexually transmitted infections prevented?",
    "What are digestive tract infections?",
    "How is a sleep disorder diagnosed?",
    "What are circulatory system issues?",
    "How does one prevent diseases?"
]

# Combine and label the data
queries = prediction_queries + non_prediction_queries
labels = [1] * len(prediction_queries) + [0] * len(non_prediction_queries)

# Create a DataFrame
df = pd.DataFrame({'query': queries, 'label': labels})

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the training data
X_train = vectorizer.fit_transform(train_df['query'])

# Transform the test data
X_test = vectorizer.transform(test_df['query'])

# Initialize the SVM model
model = SVC(kernel='linear', random_state=42)  # Using a linear kernel

# Train the model
model.fit(X_train, train_df['label'])

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(test_df['label'], y_pred)
report = classification_report(test_df['label'], y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Classification Report:\n{report}')

# Save the model and vectorizer
joblib.dump(model, 'svm_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Testing Part
# X_example = vectorizer.transform(["Can you predict the disease by taking in these symptoms?"])
# prediction = model.predict(X_example)
# print(f'Prediction: {prediction[0]}')