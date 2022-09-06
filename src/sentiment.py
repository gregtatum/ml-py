import sys
from typing import Any
from pathlib import Path
import os
import re
import shutil
import string
import tensorflow as tf

# Work around https://github.com/tensorflow/tensorflow/issues/56231
import keras.api._v2.keras as keras
from keras import layers
from keras import losses

# Tutorial from: https://www.tensorflow.org/tutorials/keras/text_classification

root_dir = (Path(__file__).parent / "..").resolve()
model_dir = root_dir / "models/sentiment"
log_dir = root_dir / "models/sentiment/logs"

print("Getting the dataset file")
dataset = keras.utils.get_file(
    fname="aclImdb_v1",
    origin="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    untar=True,
    cache_dir='data',
    cache_subdir=''
)

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')
unsup_dir = os.path.join(train_dir, 'unsup')

print("Dataset directory:", dataset_dir)
print("Remove the un-needed data: ", unsup_dir)

shutil.rmtree(unsup_dir)

batch_size = 32
seed = 0
validation_split = 0.2

print("Getting the data ready")
raw_train_ds: tf.data.Dataset = keras.utils.text_dataset_from_directory(
    'data/aclImdb/train',
    batch_size=batch_size,
    validation_split=validation_split,
    subset='training',
    seed=seed
)


raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'data/aclImdb/train',
    batch_size=batch_size,
    validation_split=validation_split,
    subset='validation',
    seed=seed
)

raw_test_ds: tf.data.Dataset = tf.keras.utils.text_dataset_from_directory(
    'data/aclImdb/test',
    batch_size=batch_size)


def standardize_text(text: str) -> str:
    lowercase = tf.strings.lower(text)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')

    s: str = tf.strings.regex_replace(
        stripped_html,
        '[%s]' % re.escape(string.punctuation),
        ''
    )
    return s


max_features = 10000

print("Create the vectorize layer")
vectorize_layer = layers.TextVectorization(
    standardize=standardize_text,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=250
)

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


def vectorize_text(text: str, label: str) -> Any:
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


print("\n")
print("Example encoding:")

text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("\n")

print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

embedding_dim = 16

model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)]
)

model.compile(
    loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0)
)

model.summary()

print("Log dir")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)


epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    # callbacks=[tensorboard_callback],
)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
