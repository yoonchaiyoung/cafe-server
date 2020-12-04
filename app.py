import tensorflow as tf
import pickle

from flask import Flask, request, jsonify
from konlpy.tag import Okt
from hanspell import spell_checker

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # CORS (cross origin error)를 방지하게 해줌  # REACT 사용하지 않으면 CORS 고려할 필요 X

NUM_WORDS = 2000

# 토크나이저 불러오기
with open('./utils/tokenizer.pickle', 'rb') as handle:  # 파이썬 객체 -> pickle로 저장이 됨
    tokenizer = pickle.load(handle)

# Sequential 형태가 아닌 일반 Model 클래스를 상속해서 만든 커스텀 클래스이다.
# .h5는 Keras의 Sequential 모델을 위한 저장 포멧이다.
# 커스텀 클래스를 이용해 모델을 만들었다면 save_weights 함수를 이용해
# 가중치만을 저장해야 한다.

# 따라서 아래 3개의 클래스(Encoder, Decoder, Seq2seq)는 따로 패키지화가 필요하다.
class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.emb = tf.keras.layers.Embedding(NUM_WORDS, 64)
        self.lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)

    def call(self, x, training=False, mask=None):
        x = self.emb(x)
        H, h, c = self.lstm(x)
        return H, h, c


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.emb = tf.keras.layers.Embedding(NUM_WORDS, 64)
        self.lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
        self.att = tf.keras.layers.Attention()
        self.dense = tf.keras.layers.Dense(NUM_WORDS, activation='softmax')

    def call(self, inputs, training=False, mask=None):
        x, s0, c0, H = inputs
        x = self.emb(x)
        S, h, c = self.lstm(x, initial_state=[s0, c0])

        S_ = tf.concat([s0[:, tf.newaxis, :], S[:, :-1, :]], axis=1)
        A = self.att([S_, H])
        y = tf.concat([S, A], axis=-1)

        return self.dense(y), h, c


class Seq2seq(tf.keras.Model):
    def __init__(self, sos, eos):
        super(Seq2seq, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()
        self.sos = sos
        self.eos = eos

    def call(self, inputs, training=False, mask=None):
        if training is True:
            x, y = inputs
            H, h, c = self.enc(x)
            y, _, _ = self.dec((y, h, c, H))
            return y
        else:
            x = inputs
            H, h, c = self.enc(x)

            y = tf.convert_to_tensor(self.sos)
            y = tf.reshape(y, (1, 1))

            seq = tf.TensorArray(tf.int32, 64)

            for idx in tf.range(64):
                y, h, c = self.dec([y, h, c, H])
                y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.int32)
                y = tf.reshape(y, (1, 1))
                seq = seq.write(idx, y)

                if y == self.eos:
                    break

            return tf.reshape(seq.stack(), (1, 64))

# 클래스 정보를 토대로 가중치가 아무것도 없는 모델을 만든다.
model = Seq2seq(sos=tokenizer.word_index['\t'],
                eos=tokenizer.word_index['\n'])

# 미리 저장해놓은 가중치를 불러온다.
model.load_weights("./utils/pretrained_ckpt")


def process_q(sentence_q):
    okt = Okt()
    seq = [' '.join(okt.morphs(sentence_q))]
    test_q_seq = tokenizer.texts_to_sequences(seq)
    test_q_padded = tf.keras.preprocessing.sequence.pad_sequences(test_q_seq,
                                                                  value=0,
                                                                  padding='pre',
                                                                  maxlen=64)

    return test_q_padded


def make_prediction(sentence_q_padded):
    answer = model.predict(sentence_q_padded)
    return spell_checker.check(tokenizer.sequences_to_texts(answer)[0]).checked


@app.route("/", methods=["GET"])
def home():
    return "Home!"


@app.route("/user_query", methods=["POST"])
def input_sentence():

    sentence_q = request.form['sentence_q']
    sentence_q_padded = process_q(sentence_q)

    return jsonify({"result": make_prediction(sentence_q_padded)})


if __name__ == "__main__":
    print("Server Start!")
    app.run("0.0.0.0", port=5000, debug=True)
