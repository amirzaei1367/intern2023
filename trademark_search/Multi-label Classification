import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPooling1D, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence

# Read the CSV file
df = pd.read_csv(r"C:\Users\nicholas.carter\Downloads\df_design_search_ml_cleaned_so.csv")

# Select the text and label columns for multi-label classification
texts = df['td_desc'].tolist()  # Replace 'td_desc' with the actual column name containing the text
labels = pd.get_dummies(df[['h3_cd']].astype(str), prefix='', prefix_sep=',')

# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to have the same length
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Create the model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 150

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(GlobalMaxPooling1D())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))  # Adding a dropout layer with dropout rate of 0.2
model.add(Dense(labels.shape[1], activation='sigmoid'))  # Use 'sigmoid' activation for multi-label classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

# Create a custom data generator using Sequence
class DataGenerator(Sequence):
    def __init__(self, x_data, y_data, batch_size):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x_data) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        return self.x_data[start:end], self.y_data[start:end]

# Create data generators
train_generator = DataGenerator(X_train, y_train, batch_size=16)
test_generator = DataGenerator(X_test, y_test, batch_size=16)

# Train the model and record accuracy history
history = model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=[early_stopping])

# Plot accuracy history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print("Loss:", loss)
print("Accuracy:", accuracy)
