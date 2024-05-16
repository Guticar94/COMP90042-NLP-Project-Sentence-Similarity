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
    shared_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.3, kernel_regularizer=l2(l2_lambda)))

    # LSTM processing for both inputs
    claim_lstm = shared_lstm(claim_embeddings)
    evidence_lstm = shared_lstm(evidence_embeddings)

    # Concatenate the outputs from LSTM layers
    concatenated = Concatenate()([claim_lstm, evidence_lstm])
    concatenated = Dropout(0.5)(concatenated)  # Increased dropout

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

# Main Execution Block
if __name__ == '__main__':
    
    # Hyper perameters
    vocab_size = 10000  # Adjust based on tokenizer knowledge
    embedding_dim = 128
    max_length = 258  # As defined in pad_sequences
    lstm_units = 128
    num_classes = 4
    learning_rate=1e-6
    l2_lambda=0.01
    batch_size=32
    epochs=13
    validation_split=0.2

    # not being used at this stage
    dropout=0.5
    activation='softmax'
    loss='categorical_crossentropy'
    metrics=['accuracy']
    
    # Load and preprocess data
    claims_data = load_json_data('train-claims.json')
    dev_claims_data = load_json_data('dev-claims.json')
    evidences_data = load_json_data('evidence.json')
    
    # Preprocess data
    claims_text = [preprocess_text(claim['claim_text']) for claim in claims_data.values()]
    dev_claims_text = [preprocess_text(claim['claim_text']) for claim in dev_claims_data.values()]
    evidence_text = [preprocess_text(evidences_data[evidence_id]) for evidence_id in evidences_data]
    
    vocab = claims_text + dev_claims_text + evidence_text  # vocab
    _, tokenizer = tokenize_and_pad(vocab, max_len=max_length, vocab_size=vocab_size)  # Shared tokenizer for vocab

    padded_claims, _ = tokenize_and_pad(claims_text+dev_claims_text, max_len=max_length, vocab_size=vocab_size)
    padded_evidences, _ = tokenize_and_pad(evidence_text, max_len=max_length, vocab_size=vocab_size)

    # Load and prepare labels
    _claims_data = claims_data.copy()
    _claims_data.update(dev_claims_data)
    labels, label_encoder = load_and_prepare_labels(_claims_data)
    
    # build / train model
    model = bid_lstm_model(vocab_size=vocab_size, embedding_dim=embedding_dim, max_length=max_length, lstm_units=lstm_units, num_classes=num_classes, learning_rate=learning_rate, l2_lambda=l2_lambda)
    #model = lstm_model_with_attention(vocab_size=vocab_size, embedding_dim=embedding_dim, max_length=max_length, lstm_units=lstm_units, num_classes=num_classes, learning_rate=learning_rate, l2_lambda=l2_lambda)
    history = train_model(model, [padded_claims, padded_evidences], labels, batch_size, epochs, validation_split)
