import csv
import requests
from pathlib import Path
import concurrent.futures


def dataset_loading(path_to_data):
    with path_to_data.open('r', encoding='utf-8') as tsvreader:
        data = csv.reader(tsvreader, delimiter='\t')
        for row in data:
            yield row[0], row[1]

def pics_mapping(path_to_data, data_gen):
    with path_to_data.open('w', encoding='utf-8') as fr:
        for i, url, descr in data_gen:
            s = ('\t').join([str(i), url,  '\n'])
            fr.write(s)

def load_url(url, timeout=10):
    r = requests.get(url, timeout=timeout)
    if not r.status_code == 200:
        raise Exception('Return code')
    return r.content


def img_name_map(map_data):
    img_map = {}
    with map_data.open('r', encoding='utf-8') as csvreader:
        data = csv.reader(csvreader, delimiter='\t')
        for row in data:
            img_map[row[1]] = row[0]

    return img_map

# def process_img(file_to_save):
    # with file_to_save.open('a')


def download(dataset, path_to_save, names_map):
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        future_to_url = {}
        for string_number, url in dataset:

            future_to_url[executor.submit(load_url, url, 20)] = url

            while len(future_to_url) > 1000:
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url.pop(future)
                    name_img = str(names_map[url]) + '.jpg'
                    name = path_to_save / name_img
                    try:
                        data = future.result()
                        with name.open('wb') as f:
                            f.write(data)
                    except Exception as exc:
                        print('%r generated an exception: %s \nSkipe: %s' % (url, exc, name))



        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url.pop(future)
            name_img = str(names_map[url]) + '.jpg'
            name = path_to_save / name_img
            try:
                data = future.result()
                with name.open('wb') as f:
                    f.write(data)
            except Exception as exc:
                print('%r generated an exception: %s \nSkipe: %s' % (url, exc, name))


if __name__ == '__main__':
    print("Download data from source...")

    path_to_save = Path('./Google_Train/')
    img_directory = Path('./Google_dataset/images_url_train.csv')

    dataset = dataset_loading(img_directory)
    # pics_mapping(img_directory, dataset)
    # dataset = dataset_loading(img_directory)
    names_map = img_name_map(img_directory)
    download(dataset, path_to_save, names_map)

