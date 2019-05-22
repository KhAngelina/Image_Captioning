import csv
import numpy as np

from math import log
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
from pathlib import Path
from keras.preprocessing import sequence

from Prepare_text_data import clean_descriptions

from pathlib import Path

import Predict_caption


# evaluate the skill of the model
def evaluate_bleu(actual_descr, predicted_descr):
    # ref = [['the dogs are in the snow in front of fence'.split(), 'the dogs play on the snow'.split()]]
    # cand = ['dog running in the snow'.split()]
    # score = corpus_bleu(ref, cand, weights=(1.0, 0, 0, 0))

    # references = []
    # candidates = []

    bleu_scores_over_all_1 = 0
    bleu_scores_over_all_2 = 0
    bleu_scores_over_all_3 = 0
    bleu_scores_over_all_4 = 0

    all_imgs_predicted = len(predicted_descr.items())

    for img_id, captions in predicted_descr.items():

        tmp_bleu_1 = 0
        tmp_bleu_2 = 0
        tmp_bleu_3 = 0
        tmp_bleu_4 = 0

        actual_desc_list = actual_descr[img_id]
        references = [actual_desc.split() for actual_desc in actual_desc_list]

        for candidate in captions:
            candidate = candidate.split()
            tmp_bleu_1 += sentence_bleu(references, candidate, weights=(1.0, 0, 0, 0))
            tmp_bleu_2 += sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0))
            tmp_bleu_3 += sentence_bleu(references, candidate, weights=(0.3, 0.3, 0.3, 0))
            tmp_bleu_4 += sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25))

        bleu_scores_over_all_1 += (tmp_bleu_1 / len(captions))
        bleu_scores_over_all_2 += (tmp_bleu_2 / len(captions))
        bleu_scores_over_all_3 += (tmp_bleu_3 / len(captions))
        bleu_scores_over_all_4 += (tmp_bleu_4 / len(captions))

        # calculate BLEU score
    print('BLEU-1: %f' % (float(bleu_scores_over_all_1) / all_imgs_predicted))
    print('BLEU-2: %f' % (float(bleu_scores_over_all_2) / all_imgs_predicted))
    print('BLEU-3: %f' % (float(bleu_scores_over_all_3) / all_imgs_predicted))
    print('BLEU-4: %f' % (float(bleu_scores_over_all_4) / all_imgs_predicted))


def evaluate_gleu(actual_descr, predicted_descr):
    # ref = [['the dogs are in the snow in front of fence'.split(), 'the dogs play on the snow'.split()]]
    # cand = ['dog running in the snow'.split()]
    # score = corpus_bleu(ref, cand, weights=(1.0, 0, 0, 0))

    # references = []
    # candidates = []

    gleu_scores_over_all_1 = 0
    gleu_scores_over_all_2 = 0
    gleu_scores_over_all_3 = 0
    gleu_scores_over_all_4 = 0

    all_imgs_predicted = len(predicted_descr.items())

    for img_id, captions in predicted_descr.items():

        tmp_gleu_1 = 0
        tmp_gleu_2 = 0
        tmp_gleu_3 = 0
        tmp_gleu_4 = 0

        actual_desc_list = actual_descr[img_id]
        references = [actual_desc.split() for actual_desc in actual_desc_list]

        for candidate in captions:
            candidate = candidate.split()
            tmp_gleu_1 += sentence_bleu(references, candidate, weights=(1.0, 0, 0, 0))
            tmp_gleu_2 += sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0))
            tmp_gleu_3 += sentence_bleu(references, candidate, weights=(0.3, 0.3, 0.3, 0))
            tmp_gleu_4 += sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25))

        gleu_scores_over_all_1 += (tmp_gleu_1 / len(captions))
        gleu_scores_over_all_2 += (tmp_gleu_2 / len(captions))
        gleu_scores_over_all_3 += (tmp_gleu_3 / len(captions))
        gleu_scores_over_all_4 += (tmp_gleu_4 / len(captions))

        # calculate GLEU score
    print('GLEU-1: %f' % (float(gleu_scores_over_all_1) / all_imgs_predicted))
    print('GLEU-2: %f' % (float(gleu_scores_over_all_2) / all_imgs_predicted))
    print('GLEU-3: %f' % (float(gleu_scores_over_all_3) / all_imgs_predicted))
    print('GLEU-4: %f' % (float(gleu_scores_over_all_4) / all_imgs_predicted))


# Flickr8k dataset
def load_actual_desc_flickr(dataset_filename):
    # load document
    descriptions = dict()
    with dataset_filename.open('r') as rf:
        lines = rf.readlines()
        for line in lines:
            tokens = line.split(' ')
            image_id, image_desc = tokens[0], tokens[1:]
            if image_id not in descriptions:
                descriptions[image_id] = list()
                # wrap description in tokens
            desc = ' '.join(image_desc)
            # store
            descriptions[image_id].append(desc)
    return descriptions


# Google dataset
def load_actual_desc_google(dataset_filename):
    # load document
    descriptions = dict()
    with dataset_filename.open('r') as rf:
        lines = rf.readlines()
        for line in lines:
            tokens = line.split('\t')
            image_id, image_desc = tokens[0], tokens[1]
            if image_id not in descriptions:
                image_desc = image_desc.split()
                image_desc = [word.lower() for word in image_desc]
                image_desc = [word for word in image_desc if len(word) > 1]
                image_desc = [word for word in image_desc if word.isalpha()]
                image_desc = ' '.join(image_desc)

                descriptions[image_id] = image_desc

    return descriptions


def load_predicted_captions(captions_file):
    captions = dict()
    with captions_file.open('r') as rf:
        lines = rf.readlines()
        for line in lines:
            tokens = line.strip().split(',')
            image_id, image_desc = tokens[0], tokens[1:]
            image_id = Path(image_id).stem
            if image_id not in captions:
                captions[image_id] = list()
                # wrap description in tokens
            for desc in image_desc:
                captions[image_id].append(desc)
    return captions




if __name__ == '__main__':
    print('Evaluate models')

    # # Flickr8k dataset
    #
    true_descriptions_file = Path('C:\\akharche\\UserPreferenceAnalysis\\Image_Captioning\\!Results\\Flickr8k_eval\\Dataset_captions\\descriptions.txt')
    predicted_captions_file = Path('C:\\akharche\\UserPreferenceAnalysis\\Image_Captioning\\!Results\\Flickr8k_eval\\google_inception_flickr8k_testset_captions.csv')

    # Lists of descriptions
    true_predictions = load_actual_desc_flickr(true_descriptions_file)
    predicted_captions = load_predicted_captions(predicted_captions_file)

    evaluate_bleu(true_predictions, predicted_captions)


    # Google dataset
    # true_descriptions_file = Path(
    #     'C:\\Angelina_caption_generation\\!Results\\Google_eval\\Dataset_captions\\images_validation_map.csv')
    # predicted_captions_file = Path(
    #     'C:\\Angelina_caption_generation\\!Results\\Google_eval\\google_model_google_testset_captions.csv')
    #
    # # Lists of descriptions
    # true_predictions = load_actual_desc_google(true_descriptions_file)
    # print(true_predictions['0'])
    # predicted_captions = load_predicted_captions(predicted_captions_file)
    #
    # evaluate_bleu(true_predictions, predicted_captions)

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




