import streamlit as st
import nltk
import random
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def load_data_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = [eval(line.strip()) for line in lines]
    return data

text_file_path = "chattextdata.txt"  # Update with your actual file path

# Load data from the text file
documents = load_data_from_file(text_file_path)

# Preprocessing steps
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
wnl = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    words = [ps.stem(wnl.lemmatize(word, pos='v')) for word in words]  # Stemming and Lemmatization
    return ' '.join(words)

# Applying preprocessing to documents
preprocessed_documents = [(preprocess_text(text), label) for text, label in documents]

# Load the pre-trained model for text classification
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform([doc[0] for doc in preprocessed_documents])
y_train = [doc[1] for doc in preprocessed_documents]

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_tfidf, y_train)

# Building Naive Bayes model for text classification
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Load the pre-trained model for image classification
image_classifier = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
image_classifier.trainable = False  # Freeze the weights of the pre-trained model

def classify_text(text):
    preprocessed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([preprocessed_text])

    # Use both Decision Tree and Naive Bayes models for text classification
    dt_prediction = dt_classifier.predict(text_tfidf)[0]
    nb_prediction = nb_classifier.predict(text_tfidf)[0]

    return dt_prediction, nb_prediction

def classify_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = image_classifier.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    image_category = decoded_predictions[0][1]
    return image_category

def greeting(sentence):
    GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
    GREETING_RESPONSES = ["hi", "hey", "nods", "hi there", "hello", "I am glad! you are talking to me"]

    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_input, dt_result, nb_result):
    return f"\nBased on the text you provided it belongs to: \nDecision Tree Prediction: {dt_result},\nNaive Bayes Prediction: {nb_result}"


def chatbot_interaction(user_input):
    user_input = user_input.lower()

    if not st.session_state.chat_history:
        # Initial greeting
        initial_greeting = "Hello, there! I'm a chatbot. I will help classify text categories between 'technical support', 'billing', 'shipping'. If you want to exit, type Bye!"
        st.session_state.chat_history.append(("Chatbot", initial_greeting))

        # Display initial greeting
        chat_history_container = st.empty()
        for role, message in st.session_state.chat_history:
            chat_history_container.write(f"{role}: {message}")

    if user_input == 'thanks' or user_input == 'thank you':
        return "You're welcome!"

    if user_input.lower() == 'bye':
        # Exit message
        exit_message = "Goodbye! If you have more questions, feel free to ask."
        st.session_state.chat_history.append(("Chatbot", exit_message))
        st.session_state.chat_history = []  # Clear chat history
        return exit_message

    if greeting(user_input) is not None:
        return greeting(user_input)

    # Use the chatbot's response function here
    dt_result, nb_result = classify_text(user_input)
    return response(user_input, dt_result, nb_result)


# Streamlit app
st.title("Text and Image Classification Chatbot")
st.write(
    "This Streamlit app is designed to assist in classifying text categories and images. You can input text by copying and pasting online text or typing, and for images, you can drag and drop files. The chatbot utilizes NLTK for text classification and a pre-trained MobileNetV2 model for image classification."
)

st.write("### Text:\n"
    "The chatbot is capable of classifying text into different categories such as 'technical support', 'billing', or 'shipping' using NLTK for text classification."
)

st.write("### Image:\n"
    "You can drag and drop an image file here, and the app will use a pre-trained MobileNetV2 model. This model predicts the most likely class or label associated with the image. For example, dropping a restaurant menu or invoice will result in predicting the text class on the image."
)

classification_choice = st.radio("Choose Classification Method:", ["Text", "Image"])

if classification_choice == "Text":
    text_input = st.text_area("You:", value='', key="text_area")
    if st.button("Send"):
        if text_input:
            chatbot_response = chatbot_interaction(text_input)
            st.session_state.chat_history.append(("You", text_input))
            st.session_state.chat_history.append(("Chatbot", chatbot_response))
            # Display chat history
            chat_history_container = st.empty()
            for role, message in st.session_state.chat_history:
                if role == "You":
                    chat_history_container.write(f"You: {message}")
                elif role == "Chatbot":
                    chat_history_container.write(f"Chatbot: {message}")

elif classification_choice == "Image":
    image_input = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    if st.button("Classify"):
        if image_input:
            image_category = classify_image(image_input)
            st.write(f"Image Classification Prediction: {image_category}")
            st.session_state.chat_history.append(("User", "Uploaded image for classification"))
            chatbot_response = "Image classification result: " + image_category
            st.session_state.chat_history.append(("Chatbot", chatbot_response))
            # Display chat history
            chat_history_container = st.empty()
            for role, message in st.session_state.chat_history:
                if role == "You":
                    chat_history_container.write(f"You: {message}")
                elif role == "Chatbot":
                    chat_history_container.write(f"Chatbot: {message}")

st.sidebar.write(
    "### Natural Language Processing (NLP)\n\n"
    "NLP is a field of artificial intelligence that focuses on the interaction "
    "between computers and humans through natural language. It involves "
    "processing and understanding human language to perform tasks such as text "
    "classification, sentiment analysis, and language translation.\n\n"
    "### Creating Chatbots with NLP\n\n"
    "NLP plays a crucial role in creating chatbots by enabling them to "
    "understand and respond to user input in a way that mimics human conversation. "
    "It involves techniques like text processing, sentiment analysis, and intent recognition.\n\n"
    "### NLTK for Text Classification\n\n"
    "NLTK (Natural Language Toolkit) is a powerful library in Python for working "
    "with human language data. It provides easy-to-use interfaces for various tasks, "
    "including text classification. In this app, NLTK is utilized for text preprocessing "
    "and classification to categorize text into different predefined categories."
)