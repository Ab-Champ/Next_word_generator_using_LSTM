# imports
import os
import time
import numpy as np
import nltk
import tensorflow as tf
from nltk.tokenize import sent_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings

warnings.filterwarnings('ignore')

class TextGenerator:
    def __init__(self, corpus_path='corpus_text.txt', model_path='text_gen_model.h5'):
        self.corpus_path = corpus_path
        self.model_path = model_path
        self.tokenizer = Tokenizer()
        self.max_len = 0
        self.vocab_size = 0
        self.model = None
        self.word_index = {}
        self.reverse_word_index = {}

    def prepare_corpus(self):
        nltk.download('punkt')
        with open(self.corpus_path, 'r', encoding='utf-8') as file:
            corpus = file.read()
        sentences = sent_tokenize(corpus)
        self.tokenizer.fit_on_texts(sentences)
        self.word_index = self.tokenizer.word_index
        self.reverse_word_index = {index: word for word, index in self.word_index.items()}

        sequences = []
        for sentence in sentences:
            tokenized = self.tokenizer.texts_to_sequences([sentence])[0]
            for i in range(1, len(tokenized)):
                sequences.append(tokenized[:i+1])

        self.max_len = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding="pre")

        X = padded_sequences[:, :-1]
        y = padded_sequences[:, -1]
        y = to_categorical(y, num_classes=len(self.word_index) + 1)

        self.vocab_size = len(self.word_index) + 1
        return X, y

    def build_model(self, input_length):
        model = Sequential()
        model.add(Embedding(self.vocab_size, 100, input_length=input_length))
        model.add(LSTM(200))
        model.add(Dense(self.vocab_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_or_load_model(self, X, y):
        if os.path.exists(self.model_path):
            print("Loading existing model...")
            self.model = load_model(self.model_path)
        else:
            print("Training new model...")
            self.model = self.build_model(X.shape[1])
            self.model.fit(X, y, epochs=150, verbose=1)
            self.model.save(self.model_path)
        self.model.summary()

    def generate_text(self, seed_text, num_words=10):
        print(f"\n--- Starting Generation with seed: '{seed_text}' ---")
        text = seed_text
        for _ in range(num_words):
            try:
                token_list = self.tokenizer.texts_to_sequences([text])[0]
                padded = pad_sequences([token_list], maxlen=self.max_len - 1, padding='pre')
                predictions = self.model.predict(padded, verbose=0)
                predicted_index = np.argmax(predictions)
                output_word = self.reverse_word_index.get(predicted_index, "")
                if output_word:
                    text += " " + output_word
                print(text)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error during generation: {e}")
                break

def main():
    generator = TextGenerator()
    X, y = generator.prepare_corpus()
    generator.train_or_load_model(X, y)

    print("\nEnter your seed text (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    user_input = "\n".join(lines)

    generator.generate_text(user_input)

if __name__ == '__main__':
    main()
