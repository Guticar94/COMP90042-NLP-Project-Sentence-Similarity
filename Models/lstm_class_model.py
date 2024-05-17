# Regularized Version - L2

import json
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.regularizers import l2

# Functions
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def preprocess_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_pad(texts, max_len, vocab_size):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<UNK>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences, tokenizer

def load_and_prepare_labels(claims_data):
    labels = [claim['claim_label'] for claim in claims_data.values()]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    onehot_encoded = to_categorical(integer_encoded)
    return onehot_encoded, label_encoder

def train_model(model, train_data, labels, batch_size, epochs, validation_split):
    return model.fit(train_data, labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

def bid_lstm_model(vocab_size, embedding_dim, max_length, lstm_units, num_classes, learning_rate, l2_lambda):
    
    # Define the input layers
    claim_input = Input(shape=(max_length,), dtype='int32')
    evidence_input = Input(shape=(max_length,), dtype='int32')

    # Shared embedding layer for both inputs
    embedding_layer = Embedding(vocab_size, embedding_dim)

    # Embedded versions of the inputs
    claim_embeddings = embedding_layer(claim_input)
    evidence_embeddings = embedding_layer(evidence_input)

    # Shared LSTM layers with L2 regularization
    shared_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.1, recurrent_dropout=0.3, kernel_regularizer=l2(l2_lambda)))

    # LSTM processing for both inputs
    claim_lstm = shared_lstm(claim_embeddings)
    evidence_lstm = shared_lstm(evidence_embeddings)

    # Concatenate the outputs from LSTM layers
    concatenated = Concatenate()([claim_lstm, evidence_lstm])
    concatenated = Dropout(0.2)(concatenated)  # Increased dropout

    # Flattening the concatenated outputs
    flattened = tf.keras.layers.Flatten()(concatenated)

    # Output classifier layer with L2 regularization
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_lambda))(flattened)

    # Build the model
    model = Model(inputs=[claim_input, evidence_input], outputs=outputs)

    # Compile the model with a learning rate schedule or adjusted learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# attention block for lstm model
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="random_normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# Bidirectional lstm blocks with attention Model - may try simple lstm with attention
def bid_lstm_model_with_attention(vocab_size, embedding_dim, max_length, lstm_units, num_classes, learning_rate, l2_lambda, dropout, recurrent_dropout):
    
    # Define the input layers
    claim_input = Input(shape=(max_length,), dtype='int32')
    evidence_input = Input(shape=(max_length,), dtype='int32')

    # Shared embedding layer for both inputs
    embedding_layer = Embedding(vocab_size, embedding_dim)

    # Embedded versions of the inputs
    claim_embeddings = embedding_layer(claim_input)
    evidence_embeddings = embedding_layer(evidence_input)

    # Shared LSTM layers with L2 regularization
    shared_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(l2_lambda)))

    # LSTM processing for both inputs
    claim_lstm = shared_lstm(claim_embeddings)
    evidence_lstm = shared_lstm(evidence_embeddings)

    # Applying attention to the LSTM outputs
    claim_attention = Attention()(claim_lstm)
    evidence_attention = Attention()(evidence_lstm)

    # Concatenate the outputs from Attention layers
    concatenated = Concatenate()([claim_attention, evidence_attention])
    concatenated = Dropout(dropout+0.1)(concatenated)  # Increased dropout

    # Output classifier layer with L2 regularization
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_lambda))(concatenated)

    # Build the model
    model = Model(inputs=[claim_input, evidence_input], outputs=outputs)

    # Compile the model with a learning rate schedule or adjusted learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Main Execution Block
if __name__ == '__main__':
     
    # Hyperparameters - in use
    vocab_size = 30000  
    max_length = 512
    embedding_dim = 128
    lstm_units = 128
    num_classes = 4
    learning_rate = 1e-3
    l2_lambda = 0.01
    batch_size = 32
    epochs = 20
    validation_split = 0.2
    
    # only being used for lstm with attention model
    dropout=0.1
    recurrent_dropout=0.3
    
    # Load and preprocess data
    training_claims_data = load_json_data('train-claims.json')
    dev_claims_data = load_json_data('dev-claims.json')
    evidences_data = load_json_data('evidence.json')
    
    # Preprocess data
    training_claims_text=[]
    training_claims_evidence=[]
    training_claims_labels=[]

    # Loading claim / evidence pairs with labels
    dev_claims_text=[]
    dev_claims_evidence=[]
    dev_claims_labels=[]

    # training claims
    for claim_id, claim_info in training_claims_data.items():
        claim_text = claim_info['claim_text']
        evidences = claim_info['evidences']
        label = claim_info['claim_label']
        for evidence_id in evidences:
            evidence_text = evidences_data[evidence_id]
            training_claims_text.append(preprocess_text(claim_text))
            training_claims_evidence.append(preprocess_text(evidence_text))
            training_claims_labels.append(label)
            
    # dev claims
    for claim_id, claim_info in dev_claims_data.items():
        claim_text = claim_info['claim_text']
        evidences = claim_info['evidences']
        label = claim_info['claim_label']
        for evidence_id in evidences:
            evidence_text = evidences_data[evidence_id]
            dev_claims_text.append(preprocess_text(claim_text))
            dev_claims_evidence.append(preprocess_text(evidence_text))
            dev_claims_labels.append(label)

    # evidence -> corpus -> tokenized corpus
    evidence_text = [preprocess_text(evidences_data[evidence_id]) for evidence_id in evidences_data]
    vocab = training_claims_text + dev_claims_text + evidence_text  # vocab
    _, tokenizer = tokenize_and_pad(vocab, max_len=max_length, vocab_size=vocab_size)  # Shared tokenizer for vocab

    # final data
    the_claims = training_claims_text + dev_claims_text
    the_evidence = training_claims_evidence + dev_claims_evidence
    _labels = training_claims_labels + dev_claims_labels

    # final data vectorised
    the_labels, _ = load_and_prepare_labels(_labels)
    padded_claims, _ = tokenize_and_pad(the_claims, max_len=max_length, vocab_size=vocab_size)
    padded_evidences, _ = tokenize_and_pad(the_evidence, max_len=max_length, vocab_size=vocab_size)
    
    # build / train model
    bi_lstm_model = bid_lstm_model(vocab_size=vocab_size, embedding_dim=embedding_dim, max_length=max_length, lstm_units=lstm_units, num_classes=num_classes, learning_rate=learning_rate, l2_lambda=l2_lambda)
    attention_lstm_model = bid_lstm_model_with_attention(vocab_size=vocab_size, embedding_dim=embedding_dim, max_length=max_length, lstm_units=lstm_units, num_classes=num_classes, learning_rate=learning_rate, l2_lambda=l2_lambda, dropout=dropout, recurrent_dropout=recurrent_dropout)
    history_bi = train_model(bi_lstm_model, [padded_claims, padded_evidences], the_labels, batch_size, epochs, validation_split)
    history_attention = train_model(attention_lstm_model, [padded_claims, padded_evidences], the_labels, batch_size, epochs, validation_split)
