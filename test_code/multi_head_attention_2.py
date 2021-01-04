import numpy as np
import pandas as pd
import tensorflow as tf
import re
import urllib.request
import tensorflow_datasets as tfds


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)

        # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
        sines = tf.math.sin(angle_rads[:, 0::2])

        # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class transformer():
    def __init__(self,
                 vocab_size=9000,
                 num_layers=4,
                 dff=512,
                 d_model=128,
                 num_heads=4,
                 dropout=0.3,
                 out_size=9000
                 ):
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dff = dff
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.out_size = out_size

        self.build()

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        # query 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        # key 크기 : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
        # value 크기 : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        # padding_mask : (batch_size, 1, 1, key의 문장 길이)

        # Q와 K의 곱. 어텐션 스코어 행렬.
        matmul_qk = tf.matmul(query, key, transpose_b=True)

        # 스케일링
        # dk의 루트값으로 나눠준다.
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)

        # 마스킹. 어텐션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
        # 매우 작은 값이므로 소프트맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
        if mask is not None:
            logits += (mask * -1e9)

        # 소프트맥스 함수는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
        # attention weight : (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
        attention_weights = tf.nn.softmax(logits, axis=-1)

        # output : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        output = tf.matmul(attention_weights, value)

        return output, attention_weights

    def split_heads(self, inputs, batch_size):
        assert self.d_model % self.num_heads == 0
        depth = self.d_model // self.num_heads

        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def multi_head_attention(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        query = tf.keras.layers.Dense(units=self.d_model)(query)
        key = tf.keras.layers.Dense(units=self.d_model)(key)
        value = tf.keras.layers.Dense(units=self.d_model)(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = self.scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # WO에 해당하는 밀집층 정의
        outpus = tf.keras.layers.Dense(units=self.d_model)(concat_attention)

        return outpus

    def encoder_layer(self):
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")

        # 인코더는 패딩 마스크 사용
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        # 멀티-헤드 어텐션 (첫번째 서브층 / 셀프 어텐션)
        attention = self.multi_head_attention({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })

        # 드롭아웃 + 잔차 연결과 층 정규화
        attention = tf.keras.layers.Dropout(rate=self.dropout)(attention)
        attention = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(inputs + attention)

        # 포지션 와이즈 피드 포워드 신경망 (두번째 서브층)
        outputs = tf.keras.layers.Dense(units=self.dff, activation='relu')(attention)
        outputs = tf.keras.layers.Dense(units=self.d_model)(outputs)

        # 드롭아웃 + 잔차 연결과 층 정규화
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention + outputs)

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs)

    def encoder(self):
        inputs = tf.keras.Input(shape=(None,), name="inputs")

        # 인코더는 패딩 마스크 사용
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        # 포지셔널 인코딩 + 드롭아웃
        embeddings = tf.keras.layers.Embedding(self.vocab_size, self.d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = PositionalEncoding(self.vocab_size, self.d_model)(embeddings)
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(embeddings)

        # 인코더를 num_layers개 쌓기
        for i in range(self.num_layers):
            outputs = self.encoder_layer()([outputs, padding_mask])

        return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs)

    def decoder_layer(self):
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")
        enc_outputs = tf.keras.Input(shape=(None, self.d_model), name="encoder_outputs")

        # 디코더는 룩어헤드 마스크(첫번째 서브층)와 패딩 마스크(두번째 서브층) 둘 다 사용.
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name="look_ahead_mask")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

        # 멀티-헤드 어텐션 (첫번째 서브층 / 마스크드 셀프 어텐션)
        attention1 = self.multi_head_attention(
            inputs={
                'query': inputs,
                'key': inputs,
                'value': inputs,  # Q = K = V
                'mask': look_ahead_mask  # 룩어헤드 마스크
            })

        # 잔차 연결과 층 정규화
        attention1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention1 + inputs)

        # 멀티-헤드 어텐션 (두번째 서브층 / 디코더-인코더 어텐션)
        attention2 = self.multi_head_attention(
            inputs={
                'query': attention1,
                'key': enc_outputs,
                'value': enc_outputs,  # Q != K = V
                'mask': padding_mask  # 패딩 마스크
            })

        # 드롭아웃 + 잔차 연결과 층 정규화
        attention2 = tf.keras.layers.Dropout(rate=self.dropout)(attention2)
        attention2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention2 + attention1)

        # 포지션 와이즈 피드 포워드 신경망 (세번째 서브층)
        outputs = tf.keras.layers.Dense(units=self.dff, activation='relu')(attention2)
        outputs = tf.keras.layers.Dense(units=self.d_model)(outputs)

        # 드롭아웃 + 잔차 연결과 층 정규화
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(outputs + attention2)

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs)

    def decoder(self):
        inputs = tf.keras.Input(shape=(None,), name='inputs')
        enc_outputs = tf.keras.Input(shape=(None, self.d_model), name='encoder_outputs')

        # 디코더는 룩어헤드 마스크(첫번째 서브층)와 패딩 마스크(두번째 서브층) 둘 다 사용.
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name='look_ahead_mask')
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

        # 포지셔널 인코딩 + 드롭아웃
        embeddings = tf.keras.layers.Embedding(self.vocab_size, self.d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = PositionalEncoding(self.vocab_size, self.d_model)(embeddings)
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(embeddings)

        # 디코더를 num_layers개 쌓기
        for i in range(self.num_layers):
            outputs = self.decoder_layer()(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs)

    def create_padding_mask(self, x):
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        # (batch_size, 1, 1, key의 문장 길이)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, x):
        seq_len = tf.shape(x)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        padding_mask = self.create_padding_mask(x)  # 패딩 마스크도 포함
        return tf.maximum(look_ahead_mask, padding_mask)

    def build_transformer(self):

        inputs = tf.keras.Input(shape=(None,), name="inputs")

        # 디코더의 입력
        dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

        # 인코더의 패딩 마스크
        enc_padding_mask = tf.keras.layers.Lambda(
            self.create_padding_mask, output_shape=(1, 1, None),
            name='enc_padding_mask')(inputs)

        # 디코더의 룩어헤드 마스크(첫번째 서브층)
        look_ahead_mask = tf.keras.layers.Lambda(
            self.create_look_ahead_mask, output_shape=(1, None, None),
            name='look_ahead_mask')(dec_inputs)

        # 디코더의 패딩 마스크(두번째 서브층)
        dec_padding_mask = tf.keras.layers.Lambda(
            self.create_padding_mask, output_shape=(1, 1, None),
            name='dec_padding_mask')(inputs)

        # 인코더의 출력은 enc_outputs. 디코더로 전달된다.
        enc_outputs = self.encoder()(inputs=[inputs, enc_padding_mask])  # 인코더의 입력은 입력 문장과 패딩 마스크

        # 디코더의 출력은 dec_outputs. 출력층으로 전달된다.
        dec_outputs = self.decoder()(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

        # 다음 단어 예측을 위한 출력층
        outputs = tf.keras.layers.Dense(units=self.out_size, name="outputs")(dec_outputs)

        return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs)

    def build(self):
        self.transformer_model = self.build_transformer()

        self.transformer_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.optimizers.Adam(lr=0.004)
        )
        self.encoder_classifier = self.encoder()
        self.encoder_classifier.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.optimizers.Adam(lr=0.004)
        )


urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData%20.csv",filename="ChatBotData.csv")
train_data = pd.read_csv('ChatBotData.csv')

questions = []
for sentence in train_data['Q']:
    # 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    questions.append(sentence)

answers = []
for sentence in train_data['A']:
    # 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    answers.append(sentence)
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2 ** 13)

# 시작 토큰과 종료 토큰에 대한 정수 부여.
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 + 2
VOCAB_SIZE = tokenizer.vocab_size + 2

MAX_LENGTH = 40


# 토큰화 / 정수 인코딩 / 시작 토큰과 종료 토큰 추가 / 패딩
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

        tokenized_inputs.append(sentence1)
        tokenized_outputs.append(sentence2)

    # 패딩
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs

questions, answers = tokenize_and_filter(questions, answers)
BATCH_SIZE = 64
BUFFER_SIZE = 20000

# 디코더의 실제값 시퀀스에서는 시작 토큰을 제거해야 한다.
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]  # 디코더의 입력. 마지막 패딩 토큰이 제거된다.
    },
    {
        'outputs': answers[:, 1:]  # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다.
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
print(dataset)
test = transformer(vocab_size=9000,
                   num_layers=4,
                   dff=512,
                   d_model=128,
                   num_heads=4,
                   dropout=0.3,
                   out_size=9000)

test.transformer_model.fit(dataset,epochs=50)

print("done")
