import pickle
import numpy as np
from math import log

from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


# extract features from photo
def extract_features(img):
    # load the model
    model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), alpha=1.0, pooling='avg', include_top=False)

    image = load_img(img, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    print(feature.shape)

    return feature


# map an integer to a word
def word_by_id(word_idx, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == word_idx:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photo_features, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    model.summary()

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word

        next_word = model.predict([photo_features, sequence], verbose=0)
        # convert probability to integer
        # TODO: BEAM_SEARCH
        next_word = argmax(next_word)

        word = word_by_id(next_word, tokenizer)

        if word is None or word == 'endseq':
            break

        in_text += ' ' + word

    return in_text.replace('startseq ', '')

# def beam_search_decoder(data, beam_size=3):
#     sequences = [[list(), 1.0]]
#     # walk over each step in sequence
#     for row in data:
#         all_candidates = list()
#         # expand each current candidate
#         for i in range(len(sequences)):
#             seq, score = sequences[i]
#             for j in range(len(row)):
#                 candidate = [seq + [j], score * -log(row[j])]
#                 all_candidates.append(candidate)
#         # order all candidates by score
#         ordered = sorted(all_candidates, key=lambda tup: tup[1])
#         # select k best
#         sequences = ordered[:beam_size]
#         print(sequences)
#     return sequences


def beam_search_predictions(model, tokenizer, photo_features, max_length, beam_index=3):
    in_text = 'startseq'
    start = tokenizer.texts_to_sequences([in_text])[0]

    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_length:
        all_candidates = []
        for s in start_word:
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            par_caps = pad_sequences([sequence], maxlen=max_length)
            next_words = model.predict([photo_features, par_caps], verbose=0)

            word_preds = np.argsort(next_words)[-beam_index:]


            # Getting the top <beam_index>(n) predictions and creating a
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0:], s[1]
                next_cap.append(w)
                prob += word_preds[0][w]
                all_candidates.append([next_cap, prob])

        start_word = all_candidates
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [word_by_id(i, tokenizer) for i in start_word]

    final_caption = []

    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption


if __name__ == '__main__':
    # load the tokenizer
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # pre-define the max sequence length (from training)
    # print(tokenizer)
    max_length = 52
    # load the model
    model = load_model('./Data/new_google_model-ep008-loss3.679-val_loss3.642.h5')
    # load and prepare the photograph
    photo_features = extract_features('./Data/4.jpg')
    # generate description
    description = generate_desc(model, tokenizer, photo_features, max_length)

    description_b_s = beam_search_predictions(model, tokenizer, photo_features, max_length, beam_index=3)

    print(description)