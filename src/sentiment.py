import sys
from typing import Any
from pathlib import Path
import os
import re
import shutil
import string
import tensorflow as tf
import pickle

# Work around https://github.com/tensorflow/tensorflow/issues/56231
import keras.api._v2.keras as keras
from keras import layers
from keras import losses

# Tutorial from: https://www.tensorflow.org/tutorials/keras/text_classification

root_dir = (Path(__file__).parent / "..").resolve()
data_dir = root_dir / "data"
model_dir = root_dir / "models/sentiment"
log_dir = root_dir / "models/sentiment/logs"
vectorizer_path = model_dir / "vectorizer.pickle"
test_dir = root_dir / "data/aclImdb/test"

if not data_dir.exists():
    print("./data did not exists, please run: mkdir ./data")
    sys.exit()

# The count of words to use for the embedding layer. Every index in the feature gets
# mapped to a corresponding randomly initialized embedding vector. This vector is then
# trained by the model which will automatically group related vocabulary
# algorithmically (I really mean magically).
max_features = 10000
embedding_dim = 16
batch_size = 32
seed = 0
validation_split = 0.2


def standardize_text(text: str) -> str:
    lowercase = tf.strings.lower(text)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')

    s: str = tf.strings.regex_replace(
        stripped_html,
        '[%s]' % re.escape(string.punctuation),
        ''
    )
    return s


vectorize_layer = layers.TextVectorization(
    standardize=standardize_text,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=250
)


def vectorize_text(text: str, label: str) -> Any:
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def get_datasets() -> Any:
    print("🚛 Loading dataset")
    dataset = keras.utils.get_file(
        fname="aclImdb_v1",
        origin="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        untar=True,
        cache_dir='data',
        cache_subdir=''
    )

    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    train_dir = os.path.join(dataset_dir, 'train')
    unsup_dir = os.path.join(train_dir, 'unsup')

    print("📁 Dataset directory:", dataset_dir)
    print("🧹 Remove the un-needed data: ", unsup_dir)

    shutil.rmtree(unsup_dir)

    print("👷 Getting the data ready")
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

    vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)

    return train_ds, val_ds


def build_model() -> keras.Sequential:
    if model_dir.exists():
        # Loading the TextVectorization
        from_disk = pickle.load(open(vectorizer_path, "rb"))
        global vectorize_layer
        vectorize_layer = layers.TextVectorization.from_config(from_disk['config'])
        # You have to call `adapt` with some dummy data (BUG in Keras)
        vectorize_layer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        vectorize_layer.set_weights(from_disk['weights'])

        print("🔎 Model found at {}".format(model_dir))
        model = keras.models.load_model(model_dir)
        print("  To re-build, run:")
        print("  rm -rf {}".format(model_dir))
        return model

    train_ds, val_ds = get_datasets()

    print("No model found, creating a new one")

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

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[tensorboard_callback],
    )

    print("💿 Saving the model to: {}".format(model_dir))
    model.save(model_dir)
    pickle_data = {
        'config': vectorize_layer.get_config(),
        'weights': vectorize_layer.get_weights()
    }
    pickle.dump(pickle_data, open(vectorizer_path, "wb"))

    analyze_model(model)
    return model


def get_test_dataset() -> tf.data.Dataset:
    raw_test_ds: tf.data.Dataset = tf.keras.utils.text_dataset_from_directory(
        'data/aclImdb/test',
        batch_size=batch_size
    )
    return raw_test_ds.map(vectorize_text)


model = build_model()


def analyze_model(model: keras.Sequential) -> None:
    loss, accuracy = model.evaluate(get_test_dataset())

    print("=================================================================")
    print("Results against test data")
    print("  Accuracy: ", accuracy)
    print("      Loss: ", loss)
    print("_________________________________________________________________")


export_model = keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

devnull = open(os.devnull, 'w')
stdout = sys.stdout

while True:
    print("")
    value = input("> Enter text to analyze the sentiment: ")

    sys.stdout = devnull
    prediction = export_model.predict([value])
    sys.stdout = stdout

    pos_ratio = prediction[0][0]
    print("")
    if pos_ratio > 0.45 and pos_ratio < 0.55:
        print("Your text has no strong sentiment. ({}% positive)".format(int(pos_ratio * 100)))
    elif pos_ratio > 0.5:
        print("Your text has a positive sentiment. ({}% positive)".format(int(pos_ratio * 100)))
    else:
        print("Your text has a negative sentiment. ({}% negative)".format(
            int(100 - pos_ratio * 100)))
