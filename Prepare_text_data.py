import string
import csv
import pickle
from pathlib import Path


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


if __name__ == '__main__':
    # filename = Path('.\\images_mapping_train.csv')
    file_to_save = Path('.\\train_clear_descr.csv')
    # load descriptions
    # dataset = load_doc(filename)
    # clean_descriptions(dataset, file_to_save)
    vocabulary = to_vocabulary(file_to_save)
    print(vocabulary)
    print('Vocabulary Size: %d' % len(vocabulary))
    pickle.dump(vocabulary, open('.\\google_train_vocabulary.pkl', 'wb'))
