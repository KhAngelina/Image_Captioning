import csv
import numpy as np
from pathlib import Path
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add


# load clean descriptions into memory
def load_clean_descriptions(filename):
    descriptions = dict()
    with filename.open('r', encoding='utf-8') as csvreader:
        data = csv.reader(csvreader, delimiter=',')
        for row in data:
            image_id = row[0]
            image_desc = row[1].split()
            descriptions[image_id] = 'startseq ' + ' '.join(image_desc) + ' endseq'
    return descriptions


# load photo features
def get_steps_by_features(filename):
    # load photo features from file
    with filename.open('r', encoding='utf-8') as csvreader:
        data = csv.reader(csvreader, delimiter='\t')
        i = 0
        for i, _ in enumerate(data):
            pass
        return i


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    all_desc = list(descriptions.values())
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_desc)
    return tokenizer


# calculate the length of the description with the most words
def max_length(descriptions):
    lines = list(descriptions.values())
    return max(len(d.split()) for d in lines)


# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, description, photo_features):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    # encode the sequence
    seq = tokenizer.texts_to_sequences([description])[0]
    # split one sequence into multiple X,y pairs
    for i in range(1, len(seq)):
        # split into input and output pair
        in_seq, out_seq = seq[:i], seq[i]
        # pad input sequence
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

        # encode output sequence
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        # store
        X1.append(photo_features)
        X2.append(in_seq)
        y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)


# define the captioning model
def define_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(1280,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model


def dataset_loading(path_to_data):
    with path_to_data.open('r', encoding='utf-8') as tsvreader:
        data = csv.reader(tsvreader, delimiter=',')
        for row in data:
            yield row[0], row[1:]


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, features_generator, tokenizer, max_length):
    # while True:
        for img_id, features in features_generator:

            description = descriptions[img_id]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, description, features)

            yield [[in_img, in_seq], out_word]



if __name__ == '__main__':
    file_train_descriptions = Path('./Data/train_clear_descr.csv')
    file_train_features = Path('./Data/google_train_features.csv')

    train_descriptions = load_clean_descriptions(file_train_descriptions)
    print('Descriptions: train=%d' % len(train_descriptions))


    # prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)

    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)

    # determine the maximum sequence length
    max_length = max_length(train_descriptions)
    print('Description Length: %d' % max_length)

    # define the model
    model = define_model(vocab_size, max_length)

    # train the model, run epochs manually and save after each epoch
    epochs = 10
    steps = get_steps_by_features(file_train_features)
    print(steps)

    for i in range(epochs):
        # create the data generator
        features_generator = dataset_loading(file_train_features)
        generator = data_generator(train_descriptions, features_generator, tokenizer, max_length)

        # fit for one epoch
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        # save model
        model.save('google_model_' + str(i) + '.h5')
