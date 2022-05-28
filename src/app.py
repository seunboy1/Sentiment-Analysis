import os
import joblib
import pymongo
import streamlit as st
from pymongo import MongoClient
from src.preprocessing import tfidf_preprocessing

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

    mylist = [
        { "name": "Amy", "address": "Apple st 652"},
        { "name": "Hannah", "address": "Mountain 21"},
        { "name": "Michael", "address": "Valley 345"},
        { "name": "Sandy", "address": "Ocean blvd 2"},
        { "name": "Betty", "address": "Green Grass 1"},
        { "name": "Richard", "address": "Sky st 331"},
        { "name": "Susan", "address": "One way 98"},
        { "name": "Vicky", "address": "Yellow Garden 2"},
        { "name": "Ben", "address": "Park Lane 38"},
        { "name": "William", "address": "Central st 954"},
        { "name": "Chuck", "address": "Main Road 989"},
        { "name": "Viola", "address": "Sideway 1633"}
    ]

    x = db.animal_tb.insert_many(mylist)

    return db

def get_stored_animals():
    db=""
    try:
        db = get_db()
        _animals = db.animal_tb.find()
        animals = [{"name": animal["name"], "address": animal["address"]} for animal in _animals]
        return animals
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
                st.write( get_stored_animals() )
            elif model_type == "LSTM Model":  
                st.markdown(inference(model_type, review, tdidf_model))
            elif model_type == "Transformer Model":  
                st.markdown(inference(model_type, review, tdidf_model))
        else:
            st.markdown('Add a review.')
