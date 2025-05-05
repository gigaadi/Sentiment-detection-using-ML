import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pickle

# Ensure you have the necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """
    Function to preprocess text data.
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    return text

def train_and_pickle_model(data_file, model_file):
    """
    Function to train an SVM model and pickle it along with the TF-IDF vectorizer.
    
    Parameters:
    - data_file: CSV file containing 'Sentences' and 'Sentiment' columns.
    - model_file: File path to save the pickled model and vectorizer.
    """
    # Load the dataset
    data = pd.read_csv(data_file)

    # Preprocess the text
    data['Sentences'] = data['Sentences'].apply(preprocess_text)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['Sentences'], data['Sentiment'], test_size=0.2, random_state=400)

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Initialize and train the SVM classifier
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_tfidf, y_train)

    # Save the model and vectorizer
    with open(model_file, 'wb') as f:
        pickle.dump((vectorizer, svm_classifier), f)

    # Predict the test data
    y_pred_svm = svm_classifier.predict(X_test_tfidf)

    # Calculate the accuracy for SVM
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print("SVM Accuracy:", accuracy_svm)

    # Print the classification report for SVM
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm))

    # Initialize and train the Random Forest classifier
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train_tfidf, y_train)

    # Predict the test data with Random Forest
    y_pred_rf = rf_classifier.predict(X_test_tfidf)

    # Calculate the accuracy for Random Forest
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print("Random Forest Accuracy:", accuracy_rf)

    # Print the classification report for Random Forest
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))

    # Return accuracies for reference (optional)
    return accuracy_svm, accuracy_rf

if __name__ == '__main__':
    # Replace 'classified.csv' and 'sentiment_model.pkl' with your file paths
    train_and_pickle_model('classified.csv', 'sentiment_model.pkl')
