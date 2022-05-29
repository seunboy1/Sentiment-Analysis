import os
import joblib
import pymongo
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from preprocessing import tfidf_preprocessing

@st.cache(allow_output_mutation=True)
def load_models():
    # This loads all the models
    tdidf_model = joblib.load('Models/model.pkl')

    return tdidf_model

@st.cache(allow_output_mutation=True)
def get_db():
    client = MongoClient(host='test_mongodb',
                         port=27017, 
                         username='root', 
                         password='pass',
                        authSource="admin")
    db = client["test_db"]
    df = pd.read_csv('data/cleaned_data.csv')
    data = df.to_dict(orient="records")
    
    x = db.data_tb.insert_many(data)

    return db

def query():
    db=""
    try:
        db = get_db()
        data = db.data_tb.find()
        result = []
        for item in data[:5]:
            result.append({"Review": item["review"], "Sentiments": item["sentiment"]})
        return result
    except:
        pass
    finally:
        if type(db)==MongoClient:
            db.close()


def inference(model_type, review, model):
    if model_type == "Tdidf Model":
        text = tfidf_preprocessing(review)
        model = load_models()
        score = model.predict([text])
        if score == 1:
            return "This is a positive review"
        else:
            return "This is a negative review"

    elif model_type == "LSTM Model":
        pass
    elif model_type == "Transformer Model":
        pass
    pass


if __name__ == "__main__":

    tdidf_model = load_models()

    #---------------------------------#
    # display title and description
    #---------------------------------#

    st.title("IMBD Sentiment Analysis")
    st.write("Displays the sentiments of any movie review")


    #---------------------------------#
    # Select model 
    #---------------------------------#
    model_type =  st.sidebar.selectbox("Select Model", ("Select", "Tdidf Model", "LSTM Model", "Transformer Model"))

    #---------------------------------#
    # display sentiment input slot
    #---------------------------------#

    review = st.text_input("Add Review", "")
    if st.button('Get Sentiment'):
        if review:
            if model_type == "Select":
                st.markdown('Please select model type')
            elif model_type == "Tdidf Model": 
                st.markdown(inference(model_type, review, tdidf_model))
                st.write( query() )
            elif model_type == "LSTM Model":  
                st.markdown(inference(model_type, review, tdidf_model))
            elif model_type == "Transformer Model":  
                st.markdown(inference(model_type, review, tdidf_model))
        else:
            st.markdown('Add a review.')
