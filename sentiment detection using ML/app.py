import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load the model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

# Streamlit app
st.title('Student Reviews Sentiment Analysis')
st.write('Enter a student review to analyze its sentiment.')

# Input text box
user_input = st.text_area("Review")

if st.button('Analyze'):
    if user_input:
        user_input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(user_input_vectorized)[0]
        st.write(f'Sentiment: {prediction}')
    else:
        st.write('Please enter a review.')
