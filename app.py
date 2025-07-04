import streamlit as st
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np

#Load the models
model = load_model("next_words_lstm.h5")

#load the tokenizer
with open('tokenizers.pickle','rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model,tokenizer,text,max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_seq_len:
        token_list = token_list[-(max_seq_len):] # Ensure sequnec length mtches max_sequence_length
    token_list = pad_sequences([token_list],maxlen = max_seq_len-1,padding='pre')
    predicted = model.predict(token_list,verbose=0)
    predicted_word_index = np.argmax(predicted,axis=1) # which ever word has thehighest probability value, take that index
    for word,index in tokenizer.word_index.items(): # convert that index to a word
        if index == predicted_word_index:
            return word
    return None

#Streamlit app

st.title("Next word Prediction with LSTM and Early Stopping")
input_text = st.text_input("Enter the sequence of words")
if st.button("Predict next word"):
    max_sequence_len = model.input.shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next word:{next_word}")
