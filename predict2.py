import re
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 加载保存的模型
model = load_model("models2/lstm_model.h5")
max_length = 200 # 序列统一长度

# 定义文本预处理函数
def clean_text(doc):
    doc = doc.replace("<br />", " ")
    doc = re.sub("<.+?>", "", doc)
    doc = re.sub("[^\w\s]", "", doc)
    doc = re.sub("\d+", "", doc)
    doc = re.sub("\s+", " ", doc).strip()
    return doc

# 加载 tokenizer
tokenizer = joblib.load('models2/tokenizer.joblib')

# 准备测试文本
test_texts = [
    'this is such an amazing movie!',
    'The movie was great!',
    'The movie was meh.',
    'The movie was bad.',
    'The movie was terrible...'
]

# 将测试文本转换为序列
test_texts_seq = tokenizer.texts_to_sequences([clean_text(text) for text in test_texts])

# 填充序列
test_texts_padded = pad_sequences(test_texts_seq, maxlen=max_length)

# 预测
predictions = model.predict(test_texts_padded)

# 输出结果
for text, pred in zip(test_texts, predictions):
    sentiment = "Positive" if pred >= 0.5 else "Negative"
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment} (Score: {pred[0]:.4f})")
    print("-" * 50)

