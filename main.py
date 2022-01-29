import imp
import pandas as pd
import gzip
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk import pos_tag
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
#defining the object for stemming
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

#reading the data
df = getDF('reviews_Office_Products_5.json.gz')
#print(df['reviewText'])
class TextPipeline:
  
  def __init__(self,df,column) :
    self.df=df
    self.column=column

  def _to_lower(self):
    df[self.column]= df[self.column].apply(lambda x: str(x).lower())
    return df

  def remove_punctuation(self):
#defining the function to remove punctuation
   def remove_txt_punctuation(text):
     for punc in string.punctuation:
       text=text.replace(punc,"")
     return text
#storing the puntuation free text
   df[self.column]= df[self.column].apply(lambda x:remove_txt_punctuation(x))  

   return df

  
#defining function for tokenization
  def tokenization(self):
#applying function to the column
   df['msg_tokenied']= df[self.column].apply(lambda x:word_tokenize(x))
   return df

  #defining the function to remove stopwords from tokenized text
  def remove_stopwords(Self):
    def remove_Txt_Stopwords(words):
     output= [i for i in words if i not in stopwords]
     return output

    df['no_stopwords']= df['msg_tokenied'].apply(lambda x:remove_Txt_Stopwords(x))
    return df

   #defining a function for stemming
  def _txtStemming(Self):
    def stemming(text):
     stem_text = [porter_stemmer.stem(word) for word in text]
     return stem_text

    df['msg_stemmed']=df['no_stopwords'].apply(lambda x: stemming(x)) 
    return df

  #defining a function for stemming
  def lemmatizer(Self):
    def _txt_lemmatizer(text):
     lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
     return lemm_text

    df['msg_lemmatized']=df['no_stopwords'].apply(lambda x:_txt_lemmatizer(x)) 
  
    return df
    
    #word frequency
  def _countWordFrequency(self):
    
    fdist1 = nltk.FreqDist(df['msg_tokenied'])
    filtered_word_freq = dict((word, freq) for word, freq in fdist1.items() if not word.isdigit()) 
    return filtered_word_freq

   # Function to generate n-grams from sentences.
  def extract_ngrams(self, num):
      n_grams = ngrams(df['msg_tokenied'], num)
      return [ ' '.join(grams) for grams in n_grams]

  def _pos(self):
   return df['msg_tokenied'].apply(lambda x:pos_tag(x))
      
      
txtpipeLine= TextPipeline(df,"reviewText")      
txtdatlower = txtpipeLine._to_lower()
txtremovePunct= txtpipeLine.remove_punctuation()
txttokenization= txtpipeLine.tokenization()   
txtRemoveStopWords= txtpipeLine.remove_stopwords()
txtcountFrequency=txtpipeLine._countWordFrequency()
txtPOS= txtpipeLine._pos()  
#print(txtdatlower["reviewText"])
#print(txttokenization)
#print(txtRemoveStopWords)
print(txtPOS)
#print(txtcountFrequency)