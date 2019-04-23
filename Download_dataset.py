import csv
import requests
import io
from pathlib import Path
from keras.preprocessing.image import load_img
from multiprocessing.dummy import Pool as ThreadPool
from keras.utils import get_file
from functools import partial
import concurrent.futures
import pickle


def dataset_loading(path_to_data):
    with path_to_data.open('r', encoding='utf-8') as tsvreader:
        data = csv.reader(tsvreader, delimiter='\t')
        for row in data:
            yield row[0], row[1]

def pics_mapping(path_to_data, data_gen):
    with path_to_data.open('w', encoding='utf-8') as fw:
        for i, url, descr in data_gen:
            s = ('\t').join([str(i), url,  '\n'])
            fw.write(s)

def load_url(url, timeout=10):
    with requests.get(url, timeout=timeout) as r:
        if not r.status_code == 200:
            raise Exception
        return r.content


def img_name_map(map_data):
    img_map = {}
    with map_data.open('r', encoding='utf-8') as csvreader:
        data = csv.reader(csvreader, delimiter='\t')
        for row in data:
            img_map[row[1]] = row[0]

    return img_map


def download(dataset, path_to_save, names_map):
    urls = []
    for string_number, url in dataset:
        urls.append(url)

        if (len(urls) % 10 == 0) or (string_number == '3318332'):
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_url = {executor.submit(load_url, url, 60): url for url in urls}
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        data = future.result()
                        name_img = str(names_map[url]) + '.jpg'
                        name = path_to_save / name_img
                        with open(name, 'wb') as f:
                            f.write(data)
                    except Exception as exc:
                        print('%r generated an exception: %s' % (url, exc))
            urls = []


if __name__ == '__main__':
    print("Download data from source...")

    path_to_save = Path('./Google_Train/')
    img_directory = Path('./Google_dataset/images_url_train.csv')

    dataset = dataset_loading(img_directory)

    # pics_mapping(img_directory, dataset)
    # dataset = dataset_loading(img_directory)
    names_map = img_name_map(img_directory)
    download(dataset, path_to_save, names_map)

