from pickle import dump
import csv
import requests
import io
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path


def dataset_loading(path_to_data):
    with path_to_data.open('r', encoding='utf-8') as tsvreader:
        data = csv.reader(tsvreader, delimiter='\t')
        for row in data:
            yield row[1]


# extract features from each photo in the directory
def extract_pic_features(pics):
    # load the model
    model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), alpha=1.0, pooling='avg', include_top=False)

    headers = {"Accept-Language": "en-US,en;q=0.5"}

    features = dict()
    for pic in pics:
        pic = 'https://www.papercitymag.com/wp-content/uploads/2017/06/Menil-40-680x1024.jpg'

        # try:
        data_img = requests.get(pic, headers=headers).content
        # print(model.summary())
        image = load_img(io.BytesIO(data_img), target_size=(224, 224))
        # except OSError:
        #     print("Skip: ", pic)
        #     continue
        image = img_to_array(image)

        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        features[pic] = feature[0]
    return features


if __name__ == '__main__':
    print("Extract features")
    # extract features from all images
    img_directory = Path('.\\Google_dataset\\Train_GCC-training.tsv')
    dataset = dataset_loading(img_directory)
    features = extract_pic_features(dataset)
    print('Extracted Features: %d' % len(features))
    # save to file
    dump(features, open('.\\google_mobilenetv2_features.pkl', 'wb'))

