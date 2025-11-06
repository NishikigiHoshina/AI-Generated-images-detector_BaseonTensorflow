
import re
import os, nltk
proj_nltk = os.path.join(os.path.dirname(__file__), 'nltk_data')  # 项目根/nltk_data
nltk.data.path.append(proj_nltk)                                  # 优先放最前面
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Download stopwords if not already present
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)
