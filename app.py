from flask import Flask, request, render_template
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        text = request.values['text']
    # 载入预训练的 tokenizer 和模型权重
    model = TFAutoModelForSequenceClassification.from_pretrained('C:/Users/jerry/OneDrive/桌面/bot_detection/flask_web/model/bert_hotel')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

    # 对输入数据进行处理和编码
    predict_text = tokenizer(text,
            add_special_tokens = True, # add [CLS], [SEP]
            padding="max_length", truncation=True, max_length=128, # 128
            return_attention_mask = True, # add attention mask to not focus on pad tokens
            return_tensors = 'tf') ###
    # 使用模型进行推理
    prediction = model(predict_text, training=False)
    prediction_probs = tf.nn.softmax(prediction.logits, axis=1).numpy()
    # 返回预测结果
    if prediction_probs[0][0] >= prediction_probs[0][1]:
        prob = ('此言論為負面言論的機率為:')
        prob_num = (str(prediction_probs[0][0]))
        video_url = "static/neg.mp4"
    else:
        prob = ('此言論為正面言論的機率為:')
        prob_num = (str(prediction_probs[0][1]))
        video_url = "static/pos.mp4"
    return render_template("predict.html", sentiment = prob, prob_num = prob_num, url = video_url)

    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)