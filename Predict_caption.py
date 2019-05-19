import pickle
import numpy as np
import pathlib
import csv

import concurrent.futures

from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model



# extract features from photo
def extract_features(img, model_features):
    # load the model

    image = load_img(img, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    features = model_features.predict(image, verbose=0)
    return features


def dataset_loading(path_to_data):
    with path_to_data.open('r', encoding='utf-8') as fr:
        data = csv.reader(fr, delimiter=',')
        for row in data:
            yield row[0], row[1:]


def features_generator(path_to_data, pictures_path, path_to_save):
    model_features = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), alpha=1.0, pooling='avg', include_top=False)
    with path_to_data.open('r', encoding='utf-8') as fr:
        data = fr.readlines()
        with path_to_save.open('w') as fw:
            for i, row in enumerate(data):
                print(i)

                row = row.strip()
                pic_info = pictures_path / row
                photo_features = extract_features(pic_info, model_features)[0]
                feature_str = (',').join([str(f) for f in photo_features])
                fw.write(row + ',' + feature_str+'\n')

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
    # model.summary()

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word

        next_word = model.predict([np.asarray(photo_features), np.asarray(sequence)], verbose=0)
        # convert probability to integer
        next_word = argmax(next_word)

        word = word_by_id(next_word, tokenizer)

        # print(str(next_word) + "    " + str(word))

        if word is None or word == 'endseq':
            break

        in_text += ' ' + word

    return in_text.replace('startseq ', '')


def process_captions(captions, tokenizer):
    result_captions = []
    end_id = tokenizer.texts_to_sequences(['endseq'])[0][0]
    for caption in captions:
        processed_caption = caption[0][1:]
        # try:
        end_index = processed_caption.index(end_id)
        processed_caption = processed_caption[:end_index]
        result_captions.append(' '.join(word_by_id(idx, tokenizer) for idx in processed_caption))
        # except Exception:
        #     pass

    return result_captions

def get_best_caption(captions, tokenizer):
    end_id = tokenizer.texts_to_sequences(['endseq'])[0][0]
    captions.sort(key=lambda l: l[1])
    best_caption = captions[-1][0]

    processed_caption = best_caption[1:]
    end_index = processed_caption.index(end_id)
    processed_caption = processed_caption[:end_index]

    return ' '.join([word_by_id(i, tokenizer) for i in processed_caption])


def beam_search_predictions(model, tokenizer, photo_features, max_length, beam_index=3):
    start = tokenizer.texts_to_sequences(['startseq'])[0]
    end = tokenizer.texts_to_sequences(['endseq'])[0][0]
    # print(photo_features.shape)

    captions = [[start, 0.0]]

    completed_captions = []

    while len(captions[0][0]) < max_length:

        all_candidates = []

        for caption in captions:
            partial_caption = pad_sequences([caption[0]], maxlen=max_length)
            next_words_pred = model.predict([np.asarray(photo_features), np.asarray(partial_caption)], verbose=0)[0]

            next_words = np.argsort(next_words_pred)[::-1][:beam_index]

            for word in next_words:
                new_partial_caption, new_partial_caption_prob = caption[0][:], caption[1]
                new_partial_caption.append(word)
                new_partial_caption_prob -= np.log(next_words_pred[word])

                # Normalization
                # new_partial_caption_prob *= 1/len(new_partial_caption)

                if word == end:
                    completed_captions.append([new_partial_caption, new_partial_caption_prob])
                else:
                    all_candidates.append([new_partial_caption, new_partial_caption_prob])

        # a = [np.exp(s[1]) for s in all_candidates]
        # print('')
        all_candidates.sort(key=lambda x: x[1]/len(x[0]))
        # all_candidates = all_candidates[::-1]
        captions = all_candidates[:beam_index]

    completed_captions.sort(key=lambda x: x[1]/len(x[0]))
    # completed_captions = completed_captions[::-1]
    res = process_captions(completed_captions[:beam_index], tokenizer)

    return res


def get_dataset_imgs(dataset_list):
    with dataset_list.open('r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines):
            yield i, line


def predict_caption(model, tokenizer, max_length, photo_features, beam_search=False, beam_size=3):

    if beam_search:
        description = beam_search_predictions(model, tokenizer, [photo_features], max_length, beam_index=beam_size)
        description = (',').join(description)
    else:
        description = generate_desc(model, tokenizer, photo_features, max_length)
    return description


def predict_captions_flickr8k(img_features_generator,
                              file_to_save,
                              model,
                              tokenizer,
                              max_length=52,
                              beam_search=False,
                              beam_size=3):

    # with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        future_to_caption = {}
        with file_to_save.open('a', encoding='utf-8') as fw:
            for img, feature in img_features_generator:
                print(img)

                feature = [float(f) for f in feature]

                # future_to_caption[executor.submit(predict_caption, model, tokenizer, max_length,
                #                                   feature, beam_search, beam_size)] = img

                # while len(future_to_caption) > 5:
                #     for future in concurrent.futures.as_completed(future_to_caption):
                #         img = future_to_caption.pop(future)
                #         descr = future.result()

                descr = predict_caption(model, tokenizer, max_length, feature, beam_search, beam_size)
                fw.write(img + ',' + descr + '\n')

                # for future in concurrent.futures.as_completed(future_to_caption):
                #     line = future_to_caption.pop(future)
                #     descr = future.result()
                #     fw.write(line + ',' + descr + '\n')


if __name__ == '__main__':
    # load the tokenizer
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # pre-define the max sequence length (from training)
    # print(tokenizer)
    max_length = 52
    # load the model
    model = load_model('./Data/new_google_model-ep010-loss3.659-val_loss3.629.h5')
    # load and prepare the photograph
    flickr8k_dataset_path = pathlib.Path('C:\\akharche\\UserPreferenceAnalysis\\Image_Captioning\\Flickr8k_Data\\Flicker8k_Dataset\\')
    flickr8k_captions_save = pathlib.Path('./!Results/Flickr8k_eval/google_model_flickr8k_testset_captions_beamsearch_3.csv')
    dataset_list = pathlib.Path('./Flickr8k_Data/Flickr_8k.testImages.txt')

    dataset_list_generator =get_dataset_imgs(dataset_list)

    path_to_save = pathlib.Path('flickr8k_test_features_mobilenet.csv')

    # features_generator(dataset_list, flickr8k_dataset_path, path_to_save)

    features_generated = dataset_loading(path_to_save)


    predict_captions_flickr8k(features_generated,
                              flickr8k_captions_save,
                              model,
                              tokenizer,
                              max_length=52,
                              beam_search=True)



    # img = './Data/11.png'
    # photo_features = extract_features(img)
    # desc = generate_desc(model, tokenizer, photo_features, max_length)
    # cap = beam_search_predictions(model, tokenizer, photo_features, max_length, beam_index=3)
    #
    # print(desc)
    # print(cap)