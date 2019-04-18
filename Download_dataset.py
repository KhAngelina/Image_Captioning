import csv
import requests
import io
from pathlib import Path
from keras.preprocessing.image import load_img
from keras.utils import get_file


def dataset_loading(path_to_data):
    with path_to_data.open('r', encoding='utf-8') as tsvreader:
        data = csv.reader(tsvreader, delimiter='\t')
        for string_number, row in enumerate(data):
            yield string_number, row[1]

def pics_mapping(path_to_data, data_gen):
    with path_to_data.open('w', encoding='utf-8') as fr:
        for i, row in data_gen:
            s = (',').join([str(i), row, '\n'])
            fr.write(s)



def download(dataset, path_to_save):
    for string_number, url in dataset:
        # print(url)
        file_name = str(string_number) + '.jpg'
        img_path = path_to_save / file_name
        try:
            data_img = requests.get(url).content
            # Check if file is ok
            load_img(io.BytesIO(data_img), target_size=(224, 224))

            with open(img_path, 'wb') as f:
                f.write( data_img)
        except Exception:
             print("Skip: ", file_name)
             continue


if __name__ == '__main__':
    print("Download data from source...")

    path_to_save = Path('C:\\Angelina_caption_generation\\Image_Captioning\\Google_Train\\')
    img_directory = Path('.\\Google_dataset\\Train_GCC-training.tsv')

    path_to_data = Path('images_mapping_train.csv')
    dataset = dataset_loading(img_directory)
    pics_mapping(path_to_data, dataset)

    # dataset = dataset_loading(img_directory)
    # download(dataset, path_to_save)