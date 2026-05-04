import nltk
import string
from nltk.corpus import stopwords

# download once
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # lowercase
    text = text.lower()
    
    # remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)