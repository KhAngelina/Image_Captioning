from pathlib import Path
import csv


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



if __name__ == '__main__':
    target_file = Path('./google_train_features.csv')

    file1 = Path('./google_train_features1.csv')
    file2 = Path('./google_test_features.csv')

    split(target_file, file1, file2)



