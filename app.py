import streamlit as st
import pickle

model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')
text=st.text_input('Enter message')

import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

def transform_text(text:str):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for word in text:
        if word.isalnum():
            y.append(word)
    
    text=y
    y=[]
    
    ps=PorterStemmer()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(ps.stem(word))

    return " ".join(y)
if st.button('Predict'):
    #1.preprocess
    text=transform_text(text)
    #2.vectorize
    text_vectorized=tfidf.transform([text])
    #3.predict
    result=model.predict(text_vectorized)[0]
    #4.display
    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')
