from os import listdir
from pickle import dump
import csv
from keras.applications.vgg16 import VGG16

from keras.applications.mobilenet_v2 import MobileNetV2

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
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

    features = dict()
    for pic in pics:
        print(model.summary())

        image = load_img(pic, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)

        features[pic] = feature
        print(feature.shape)
        print(feature)

    # # load the model
    # model = VGG16()
    # # re-structure the model
    # model.layers.pop()
    # model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # # summarize
    # print(model.summary())
    # # extract features from each photo
    # features = dict()
    # for pic in pics:
    #     # load an image from file
    #     image = load_img(pic, target_size=(224, 224))
    #     # convert the image pixels to a numpy array
    #     image = img_to_array(image)
    #     # reshape data for the model
    #     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #     # prepare the image for the VGG model
    #     image = preprocess_input(image)
    #     # get features
    #     feature = model.predict(image, verbose=0)
    #     # get image id
    #     image_id = pic
    #     # store feature
    #     features[image_id] = feature
    #     print(image_id)
    #     print(features[image_id])
    return features


if __name__ == '__main__':
    print("Extract features")
    # # extract features from all images
    # directory = 'C:\\Angelina_caption_generation\\Flickr8k_Dataset'
    #
    # photos_dir = Path('C:\\akharche\\UserPreferenceAnalysis\\user_profile')
    # imgs = get_images_all(photos_dir)
    #
    # features = extract_pic_features(imgs)
    # print('Extracted Features: %d' % len(features))
    # # save to file
    # dump(features, open('C:\\akharche\\UserPreferenceAnalysis\\data\\user_profile_vgg16_features.pkl', 'wb'))

    img_directory = Path('..\\Google_dataset\\Train_GCC-training.tsv')
    print(img_directory)

    # dataset_loading(img_directory)
    extract_pic_features(img_directory)

