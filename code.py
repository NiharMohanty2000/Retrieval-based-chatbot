import nltk
import numpy as np
import pandas as pd
import random
import string

import bs4 as bs
import requests
import re

import warnings
warnings.filterwarnings= False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')

r= requests.get('https://en.wikipedia.org/wiki/Cuisine')
raw_html = r.text

#clean_up
corpus_html= bs.BeautifulSoup(raw_html)

#to extract paragraphs
corpus_paras= corpus_html.find_all('p')
corpus_text= ''

#join paras
for para in corpus_paras:
    corpus_text += para.text

#lowercasing the text
corpus_text= corpus_text.lower()

corpus_text

#removing empty spaces and special chars
corpus_text= re.sub(r'\([0-9]*\)', ' ', corpus_text)
corpus_text= re.sub(r'\s+',' ', corpus_text)

#converting text into sentences
corpus_sentences= nltk.sent_tokenize(corpus_text)
corpus_words= nltk.word_tokenize(corpus_text)

corpus_sentences
corpus_words

#Predefined inputs
greeting_inputs= ('hey', 'good morning', 'good evening', 'whats up', 'morning', 'evening', 'hi')
greeting_responses= ['hey', 'hey hows you?', 'nods', 'hello,how you doing','hello',' hello, I am good']

def greet_response(greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)
        
#preprocessing and lemmatizing
wn_lemmatizer= nltk.stem.WordNetLemmatizer()

def lemmatize_corpus(tokens):
    return [wn_lemmatizer.lematize(tokens) for token in tokens]

punctuation_removal_dict= dict((ord(punctuation), None) for punctuation in string.punctuation)

def get_processed_text(Document):
    return lemmatize_corpus(nltk.word_tokenize(document.lower().translate(punctuation_removal_dict)))

#language modelling
def respond(user_input):
    
    bot_response= ''
    corpus_sentences.append(user_input)
    
    
    #vectorizing pre-processed text
    word_vectorizer= TfidfVectorizer(tokenizer= get_processed_text,stop_words= 'english')
    corpus_word_vectors= word_vectorizer.fit_transform(corpus_sentences)
    
    cos_sim_vectors= cosine_similarity(corpus_word_vectors[-1],corpus_word_vectors)
    similar_response_idx= cos_sim_vectors.agrsort()[0][2]
    
    matched_vector= cos_sim_vectors.flatten()
    matched_vector.sort()
    vector_matched= matched_vector[-2]
    
    if vector_matched == 0:
        bot_response= "I am sorry, what is it again?"
        return bot_response
    else:
        bot_response= bot_response + corpus_sentences[similar_response_idx]
        return bot_response
    
chat= True
print("What do you want to learn about cuisines today??")

while(chat== True):
    user_query= input()
    user_query= user_query.lower()
    if user_query != 'quit':
        if user_query == "Thanks" or user_query == "thank you":
            chat= False
            print("Cuisine bot: You're welcome")
        else:
            if greet_response(user_query) != None:
                print(" Cuisine bot: " + greet_response(user_query))
            else:
                print("Cuisine bot: ",end="")
                print(respond(user_query))
                corpus_sentences.remove(user_query)
    else:
        chat= False
        print("Cuisine bot: Good Bye")