import streamlit as st
import numpy as np
import pandas as pd
import joblib,os
import nltk
import re
from nltk.corpus import stopwords
import tensorflow
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import ftfy
import pickle as pkl
from keras.preprocessing.text import Tokenizer
import spacy
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

st.set_option('deprecation.showPyplotGlobalUse', False)

matplotlib.use('Agg')

nlp=spacy.load('en_core_web_sm')

nltk.download('stopwords')
nltk.download('punkt')

st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)



def load_prediction_models(model_path):
    loaded_model=load_model(model_path)
    loaded_model.summary()
    return loaded_model

def expandContractions(text, c_re,cList):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

def get_keys(val,my_dic):
    if val<0.5:
        val=0.0
    else:
        val=1.0
    for key,value in my_dic.items():
        if val==value:
            return key
        
def text_clean(news_text):
    
    cList = pkl.load(open('dataset/cword_dict.pkl','rb'))
    c_re = re.compile('(%s)' % '|'.join(cList.keys()))
    news=news_text
    if re.match("(\w+:\/\/\S+)", news) == None and len(news) > 5:
            news = ' '.join(re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", news).split())
            news = ftfy.fix_text(news)
            news = expandContractions(news,c_re,cList)
            news = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", news).split())
            stop_words = stopwords.words('english')
            word_tokens = nltk.word_tokenize(news) 
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            news = ' '.join(filtered_sentence)
            news = PorterStemmer().stem(news)
    return news

def text_embedding(review):
    nb_max_words=100000
    max_length=300 
    
    one_hot_repr=[one_hot(review,nb_max_words)]
    embedded_docs=pad_sequences(one_hot_repr,padding='pre',maxlen=max_length)
    embedded_array=np.array(embedded_docs)
    
    return embedded_array
    
        
def main():
    """Fake News Classifier with Streamlit"""
    st.title('Fake News Classifier')
    st.subheader('Deep Learning App with Streamlit')
    
    activities=['Prediction','NLP']
    
    choice=st.sidebar.selectbox('Choose Activity',activities)
    
    if choice=='Prediction':
        st.info('Prediction with Deep Learning Models')
        
        news_text=st.text_area('Enter Tweet','Type Here')
        all_dl_models=['CNN-LSTM Model','LSTM Model','CNN Model','Bidirectional-LSTM Model']
        model_choice=st.selectbox('Choose DL Model',all_dl_models)
        prediction_labels={'True':0,'Fake':1}
        
        if st.button('Predict'):
            if len(news_text)<=250:
                st.info('Text length too small to predict. Try again.')
            else:
                st.text('Original text :: \n{}'.format(news_text))
                review=text_clean(news_text)
                #print(review)
                embedding=text_embedding(review)
                #print(embedding)
                if model_choice=='CNN Model':
                    predictor=load_prediction_models('models/CNN_model (1).h5')
                    prediction=predictor.predict(embedding)
                    #st.write(prediction)
                    final_result=get_keys(prediction,prediction_labels)
                    st.success("News categorized as : : {}".format(final_result))
                
                if model_choice=='LSTM Model':
                    predictor=load_prediction_models('models/LSTM_model (1).h5')
                    prediction=predictor.predict(embedding)
                    #st.write(prediction)
                    final_result=get_keys(prediction,prediction_labels)
                    st.success("News categorized as : : {}".format(final_result))
                
                if model_choice=='CNN-LSTM Model':
                    predictor=load_prediction_models('models/CNN-LSTM_model (1).h5')
                    prediction=predictor.predict(embedding)
                    #st.write(prediction)
                    final_result=get_keys(prediction,prediction_labels)
                    st.success("News categorized as : : {}".format(final_result))
                
                if model_choice=='Bidirectional-LSTM Model':
                    predictor=load_prediction_models('models/BiLSTM_model (1).h5')
                    prediction=predictor.predict(embedding)
                    final_result=get_keys(prediction,prediction_labels)
                    st.success('News categorized as : : {}'.format(final_result))
            
            
  
    if choice=='NLP':
        st.info('Deep Natural Language Processing')
        news_text=st.text_area("Enter text","Type Here")
        nlp_task=['Tokenization','NER','Lemmatization','POS Tags']
        task_choice=st.selectbox('Choose NLP Task',nlp_task)
        
        if st.button('Analyze'):
            st.info('Original text : : {}'.format(news_text))
            docx=nlp(news_text)
            if task_choice=='Tokenization':
                result=[token.text for token in docx]
                
                
            elif task_choice=='Lemmatization':
                result=["'Token' : {}, 'Lemma' : {}".format(token.text,token.lemma_) for token in docx]
            
            elif task_choice=='NER':
                result=[(entity.text,entity.label_) for entity in docx.ents]
                
            elif task_choice=='POS Tags':
                result=["'Token' : {}, 'POS' : {}, 'Dependency' : {}".format(word.text,word.tag_,word.dep_) for word in docx]
                
            
            st.json(result)
            
            
        if st.button('Tabulize'):
            docx=nlp(news_text)
            c_tokens=[token.text for token in docx]
            c_lemma=[(token.lemma_) for token in docx]
            c_pos=[(word.tag_) for word in docx]
                
                
            new_df=pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns=['Tokens','Lemma','POS'])
            st.dataframe(new_df)
            
        
        if st.button('Wordcloud'):
            
            wordcloud=WordCloud().generate(news_text)
            plt.imshow(wordcloud,interpolation='bilinear')
            st.pyplot()
            
                
                
            
    
if __name__=='__main__':
    main()


