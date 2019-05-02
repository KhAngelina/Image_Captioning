import csv
import numpy as np
from math import log
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from pathlib import Path


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


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
def load_photo_features(filename):
    # load photo features from file
    with filename.open('r', encoding='utf-8') as csvreader:
        data = csv.reader(csvreader, delimiter=',')
        features = {row[0]: row[1:] for row in data}

        return features


# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    all_desc = list(descriptions.values())
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_desc)
    return tokenizer


# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def beam_search_decoder(data, k=5):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences

# generate a description for an image
def generate_desc(model, tokenizer, photo_features, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo_features, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


def get_desc(model, pics, tokenizer, max_length, file_to_save):
    predicted = list()
    with file_to_save.open('w', encoding='utf-8') as f:

        for img_name, features in pics.items():
            # generate description
            yhat = generate_desc(model, tokenizer, features, max_length)
            print(yhat)
            f.write((',').join([img_name, yhat]))
            f.write('\n')
                    # yield filename, sentence


# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        print(references)
        actual.append(references)
        predicted.append(yhat.split())
        print(yhat)
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


def beam_search_decoder(data, k=2):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
        print(sequences)
    return sequences

if __name__ == '__main__':
    # # prepare tokenizer on train set
#     #
#     # # load training dataset (6K)
#     # clean_descriptions_train = Path('./Data/train_clear_descr.csv')
#     # train_descriptions = load_clean_descriptions(clean_descriptions_train)
#     # # prepare tokenizer
#     # tokenizer = create_tokenizer(train_descriptions)
#     # max_length = max_length(train_descriptions)
#     # # print('Description Length: %d' % max_length)
#     #
#     # # prepare test set
#     #
#     # # load test set
#     # clean_descriptions_test = Path('./Test_data/test_clear_descr.csv')
#     # test_descriptions = load_clean_descriptions(clean_descriptions_test)
#     # print('Descriptions: test=%d' % len(test_descriptions))
#     #
#     #
#     # # photo features
#     # filename = Path('./Test_data/test_features.csv')
#     # features = load_photo_features(filename)
#     # print('Photos: test=%d' % len(features))
#     #
#     #
#     # # load the model
#     # filename = 'google_model_11.h5'
#     # model = load_model(filename)
#     # # evaluate model
#     # file_to_save = Path('./Google_Test/google_test_predictions.csv')
#     # get_desc(model, features, tokenizer, max_length, file_to_save)

    data = [[0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5]]

    beam_search_decoder(data)



