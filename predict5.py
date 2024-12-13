import re
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_files
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy

#=====================
# 配置与路径（请根据实际情况修改）
#=====================
logistic_model_path = "models/logistic_model.joblib"
vectorizer_path = "models/tfidf_vectorizer.joblib"
lstm_model_path = "models2/lstm_model.h5"
tokenizer_path = "models2/tokenizer.joblib"
max_length = 200  # 与训练LSTM模型时使用的序列长度一致
dataset_path = "review_polarity/txt_sentoken"  # 你的数据集路径

# 加载spaCy模型（用于Logistic回归的分词）
en_nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

#=====================
# 文本清理与分词函数
#=====================
def clean_text(doc):
    doc = doc.replace("<br />", " ")
    doc = re.sub("<.+?>", "", doc)
    doc = re.sub("[^\w\s]", "", doc)
    doc = re.sub("\d+", "", doc)
    doc = re.sub("\s+", " ", doc).strip()
    return doc

def tokenizer_spacy(doc):
    doc = doc.decode('utf-8') if isinstance(doc, bytes) else doc
    doc_spacy = en_nlp(doc)
    return [token.lemma_ for token in doc_spacy if not token.is_stop and not token.is_punct]

#=====================
# 加载数据集
#=====================
print("Loading validation data...")
reviews_val = load_files(dataset_path, categories=["pos", "neg"])
X_val, y_val = reviews_val.data, reviews_val.target

X_val = [clean_text(doc.decode('utf-8')) for doc in X_val]
print("Validation data loaded.")

#=====================
# 加载Logistic回归模型与TF-IDF向量器，对验证集预测
#=====================
print("\nLoading Logistic Regression model and vectorizer...")
lr_model = joblib.load(logistic_model_path)
tfidf_vectorizer = joblib.load(vectorizer_path)
print("Logistic model and vectorizer loaded.")

X_val_tfidf = tfidf_vectorizer.transform(X_val)
y_proba_lr_val = lr_model.predict_proba(X_val_tfidf)
# y_proba_lr_val为 [n_samples, 2], 第0列为负类概率, 第1列为正类概率
p_pos_lr_val = y_proba_lr_val[:, 1]

#=====================
# 加载LSTM模型与tokenizer，对验证集预测
#=====================
print("\nLoading LSTM model and tokenizer...")
lstm_model = load_model(lstm_model_path)
tokenizer = joblib.load(tokenizer_path)
print("LSTM model and tokenizer loaded.")

X_val_seq = tokenizer.texts_to_sequences(X_val)
X_val_padded = pad_sequences(X_val_seq, maxlen=max_length)
y_proba_lstm_val = lstm_model.predict(X_val_padded)  
# 输出形如 (n_samples, 1)，为正类概率
p_pos_lstm_val = y_proba_lstm_val[:, 0]

#=====================
# 在验证集上搜索最佳权重
#=====================
best_w_lr = None
best_accuracy = 0.0

print("\nSearching best weight on validation set...")
for w_lr in np.arange(0, 1.05, 0.05):
    w_lstm = 1.0 - w_lr
    p_pos_ensemble_val = w_lr * p_pos_lr_val + w_lstm * p_pos_lstm_val
    y_pred_ensemble_val = (p_pos_ensemble_val >= 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred_ensemble_val)
    if acc > best_accuracy:
        best_accuracy = acc
        best_w_lr = w_lr

best_w_lstm = 1.0 - best_w_lr

print("\nBest weight found on validation set:")
print(f"w_lr = {best_w_lr:.2f}, w_lstm = {best_w_lstm:.2f}")
print(f"Validation Accuracy = {best_accuracy:.4f}")
