import csv
import numpy as np
import math
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
from keras.callbacks import ModelCheckpoint



class TrainGenerator(object):
    """Acts as an adapter of `Dataset` for Keras' `fit_generator` method."""
    def __init__(self,
                 batch_size,
                 photo_features_train,
                 photo_features_test,
                 tokenizer,
                 captions,
                 max_length,
                 vocab_size):

        self._batch_size = batch_size
        self._photo_features_train = photo_features_train
        self._photo_features_test = photo_features_test
        self._tokenizer = tokenizer
        self._captions = captions
        self._max_length = max_length
        self._vocab_size = vocab_size

    def generate_data(self, train=True):
        X1, X2, y = list(), list(), list()

        n = 0

        photo_features = self._photo_features_train if train else self._photo_features_test

        while True:
            for img_id, features in photo_features:
                description = self._captions[img_id]

                seq = self._tokenizer.texts_to_sequences([description])[0]
                # split one sequence into multiple X,y pairs

                n += 1
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=self._max_length)[0]

                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=self._vocab_size)[0]
                    # store
                    X1.append(features)
                    X2.append(in_seq)
                    y.append(out_seq)

                if n == self._batch_size:
                    yield [[np.array(X1), np.array(X2)], np.array(y)]
                    X1, X2, y = list(), list(), list()
                    n = 0

    # define the captioning model
    def define_model(self):
        # feature extractor model
        inputs1 = Input(shape=(1280,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)

        # sequence model
        inputs2 = Input(shape=(self._max_length,))
        se1 = Embedding(self._vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        # decoder model
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self._vocab_size, activation='softmax')(decoder2)
        # tie it together [image, seq] [word]
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        # compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        # summarize model
        model.summary()
        # plot_model(model, to_file='model.png', show_shapes=True)
        return model


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


def dataset_loading(path_to_data):
    with path_to_data.open('r', encoding='utf-8') as tsvreader:
        data = csv.reader(tsvreader, delimiter=',')
        for row in data:
            yield row[0], row[1:]




if __name__ == '__main__':
    file_train_descriptions = Path('./Data/train_clear_descr.csv')
    file_train_features = Path('./Data/google_train_features.csv')

    file_test_features = Path('./Data/google_test_features.csv')

    train_descriptions = load_clean_descriptions(file_train_descriptions)
    print('Descriptions: train=%d' % len(train_descriptions))


    # prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)

    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)

    # determine the maximum sequence length
    max_length = max_length(train_descriptions)
    print('Description Length: %d' % max_length)

    features_generator_train = dataset_loading(file_train_features)
    features_generator_test = dataset_loading(file_test_features)

    batch_size = 128
    trainGenerator = TrainGenerator(batch_size=batch_size,
                                    photo_features_train=features_generator_train,
                                    photo_features_test=features_generator_test,
                                    tokenizer=tokenizer,
                                    captions=train_descriptions,
                                    max_length=max_length,
                                    vocab_size=vocab_size)

    # define the model
    model = trainGenerator.define_model()

    # train the model, run epochs manually and save after each epoch
    epochs = 10
    steps_train = math.ceil(get_steps_by_features(file_train_features) / batch_size)
    steps_test = math.ceil(get_steps_by_features(file_test_features) / batch_size)

    filepath = 'google_model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit_generator(trainGenerator.generate_data(), steps_per_epoch=steps_train, epochs=20, verbose=2, callbacks=[checkpoint],
                        validation_data=trainGenerator.generate_data(train=False), validation_steps=steps_test)
    #
    # for i in range(epochs):
    #     # create the data generator
    #     features_generator = dataset_loading(file_train_features)
    #     generator = data_generator(train_descriptions, features_generator, tokenizer, max_length)
    #
    #     # fit for one epoch
    #     model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    #     # save model
    #     model.save('google_model_' + str(i) + '.h5')



