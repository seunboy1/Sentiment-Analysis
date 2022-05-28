import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet') 
nltk.download('omw-1.4') 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def remove_html_tags(text):
    text=re.sub('<.*?>',' ',text)
    return text

def remove_special_characters(text):
    pattern=r'[^a-zA-z0-9\s]'
    text = re.sub(pattern,'',text)
    text = re.sub('\[[^]]*\]', '', text)
    return text

def to_lowercase(text):
    return text.lower()

def unify_whitespaces(text):  
    text = re.sub(" +", " ", text)
    return text

def remove_stopwords(text):
    text = text.split()
    text = [word for word in text if not word in set(stopwords.words('english'))]
    return text

def text_lemmatizing(text):
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text]
    return ' '.join(text)

def tfidf_preprocessing(text):
    text =remove_html_tags(text)
    text = remove_special_characters(text)
    text = to_lowercase(text)
    text = unify_whitespaces(text)
    text = remove_stopwords(text)
    text = text_lemmatizing(text)
    return text