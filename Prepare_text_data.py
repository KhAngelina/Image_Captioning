import string
import csv
import pickle
from pathlib import Path
import json
import numpy as np

# load doc into memory
def load_doc(filename):
    with filename.open('r', encoding='utf-8') as csvreader:
        data = csv.reader(csvreader, delimiter='\t')
        for row in data:
            print(row[0], ' ', row[1])
            yield row[0], row[1]


def clean_descriptions(descriptions, file_to_save):
    # ''-not change only delete the symbols that in string.punctuation
    table = str.maketrans('', '', string.punctuation)

    with file_to_save.open('w', encoding='utf-8') as fw:
        for image_id, image_desc in descriptions:
                # tokenize
                desc = image_desc.split()
                # convert to lower case
                desc = [word.lower() for word in desc]
                # remove punctuation from each token
                desc = [w.translate(table) for w in desc]
                # remove hanging 's' and 'a'
                desc = [word for word in desc if len(word) > 1]
                # remove tokens with numbers in them
                desc = [word for word in desc if word.isalpha()]
                # store as string
                desc = ' '.join(desc)
                fw.write(','.join([image_id, desc, '\n']))

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(clean_descr_file):
    # build a list of all description strings
    all_desc = set()
    with clean_descr_file.open('r', encoding='utf-8') as csvreader:
        data = csv.reader(csvreader, delimiter=',')
        for row in data:
            all_desc.update(row[1].split())
    return all_desc


# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def to_json_format(file_to_format, file_to_save):
    data = []

    with file_to_format.open('r') as fr:
        lines = fr.readlines()

        for line in lines:
            predictions = line.split(',')
            img_id, captions = predictions[0], predictions[1:]

            img_id = int(img_id.split('_')[-1])

            # for caption in captions:
            candidate = {'image_id': img_id, 'caption': captions[0].strip()}
            data.append(candidate)

    with file_to_save.open('w') as outfile:
        json.dump(data, outfile)

if __name__ == '__main__':
    # filename = Path('.\\images_mapping_train.csv')
    # file_to_save = Path('.\\train_clear_descr.csv')
    # # load descriptions
    # # dataset = load_doc(filename)
    # # clean_descriptions(dataset, file_to_save)
    # vocabulary = to_vocabulary(file_to_save)
    # print(vocabulary)
    # print('Vocabulary Size: %d' % len(vocabulary))
    # pickle.dump(vocabulary, open('.\\google_train_vocabulary.pkl', 'wb'))

    file_to_format = Path('C:\\akharche\\UserPreferenceAnalysis\\Image_Captioning\\!Results\\MSCOCO_eval\\google_inception_coco_testset_captions.csv')
    file_to_save = Path('C:\\akharche\\UserPreferenceAnalysis\\Image_Captioning\\!Results\\MSCOCO_eval\\google_inception_coco_testset_captions.json')

    to_json_format(file_to_format, file_to_save)


    b_1 = [0.258, 0.273, 0.28]
    b_2 = [0.112, 0.114, 0.116]
    b_3 = [0.049, 0.48, 0.051]
    b_4 = [0.02, 0.017, 0.02]
    m = [0.083, 0.084, 0.085]
    rl = [0.196, 0.2, 0.201]
    c = [0.143, 0.148, 0.141]

    print(np.sum(b_1) / len(b_1))
    print(np.sum(b_2) / len(b_1))
    print(np.sum(b_3) / len(b_1))
    print(np.sum(b_4) / len(b_1))
    print(np.sum(m) / len(b_1))
    print(np.sum(rl) / len(b_1))
    print(np.sum(c) / len(b_1))

    b_1 = [0.258, 0.273, 0.278, 0.282, 0.282]
    b_2 = [0.114, 0.117, 0.118, 0.122, 0.121]
    b_3 = [0.052, 0.051, 0.052, 0.052, 0.053]
    b_4 = [0.023, 0.02, 0.019, 0.019, 0.021]
    m = [0.082, 0.085, 0.087, 0.088, 0.088]
    rl = [0.195, 0.202, 0.201, 0.204, 0.203]
    c = [0.142, 0.15, 0.145, 0.147, 0.14]

    print("B_S 5")


    print(np.sum(b_1)/len(b_1))
    print(np.sum(b_2) / len(b_1))
    print(np.sum(b_3) / len(b_1))
    print(np.sum(b_4) / len(b_1))
    print(np.sum(m) / len(b_1))
    print(np.sum(rl) / len(b_1))
    print(np.sum(c) / len(b_1))
