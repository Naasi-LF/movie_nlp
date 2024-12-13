import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_files
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import re
import spacy

en_nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# 定义自定义分词器，使用 spaCy 进行 lemmatization
def tokenizer_spacy(doc):
    # 将字节转换为字符串
    doc = doc.decode('utf-8') if isinstance(doc, bytes) else doc
    doc_spacy = en_nlp(doc)
    return [token.lemma_ for token in doc_spacy if not token.is_stop and not token.is_punct]

# 定义文本预处理函数
def clean_text(doc):
    doc = doc.replace("<br />", " ")
    doc = re.sub("<.+?>", "", doc)
    doc = re.sub("[^\w\s]", "", doc)
    doc = re.sub("\d+", "", doc)
    doc = re.sub("\s+", " ", doc).strip()
    return doc

# 修改 TfidfVectorizer，使用自定义的 tokenizer
tfidf_spacy = TfidfVectorizer(tokenizer=tokenizer_spacy, min_df=6, stop_words="english", ngram_range=(1, 2))

def load_models_and_predict(texts, model_path="models/logistic_model.joblib", 
                          vectorizer_path="models/tfidf_vectorizer.joblib"):
    # 加载模型和向量器
    print("Loading models...")
    loaded_model = joblib.load(model_path)
    tfidf_spacy = joblib.load(vectorizer_path)
    print("Models loaded successfully!")
    
    # 转换文本
    print("\nTransforming texts...")
    texts = [clean_text(text) for text in texts]
    test_texts_tfidf = tfidf_spacy.transform(texts)
    
    # 预测
    print("Making predictions...\n")
    y_pred = loaded_model.predict(test_texts_tfidf)
    y_pred_proba = loaded_model.predict_proba(test_texts_tfidf)
    
    # 输出预测结果
    print("Prediction Results:")
    print("=" * 70)
    for text, pred, proba in zip(texts, y_pred, y_pred_proba):
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = max(proba)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probability Distribution: Negative: {proba[0]:.4f}, Positive: {proba[1]:.4f}")
        print("-" * 70)
    
    return y_pred, y_pred_proba

test_texts = [
    'this is such an amazing movie!',
    'The movie was great!',
    'The movie was meh.',
    'The movie was bad.',
    'The movie was terrible...'
]

# 进行预测
predictions, probabilities = load_models_and_predict(test_texts)

