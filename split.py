from pathlib import Path
import csv
import math


def split(target_file, file1, file2, nrow=3000000):

    with target_file.open('r') as fr:
        data = fr.readlines()

        for i, row in enumerate(data):
            if i >= nrow:
                with file2.open('a', encoding='utf-8') as fw:
                    fw.write(row)
            else:
                with file1.open('a', encoding='utf-8') as fw:
                    fw.write(row)


# load photo features
def get_row_count(filename):
    # load photo features from file
    with filename.open('r', encoding='utf-8') as csvreader:
        data = csv.reader(csvreader, delimiter='\t')
        i = 0
        for i, _ in enumerate(data):
            pass
        return i

if __name__ == '__main__':
    # target_file = Path('./google_train_features.csv')

    # file1 = Path('./google_train_features1.csv')
    # file2 = Path('./google_test_features.csv')

    # split(target_file, file1, file2)

    target_file = Path('./InseptionV3_model/google_train_features_inception.csv')

    file1 = Path('./InseptionV3_model/google_train_features_inception1.csv')
    file2 = Path('./InseptionV3_model/google_test_features_inception.csv')

    row_num = get_row_count(target_file)

    train_row_num = math.ceil(0.8*row_num)
    print(train_row_num)

    split(target_file, file1, file2, nrow=train_row_num)



