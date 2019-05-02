import csv
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
def extract_pic_features(pics_data_file, pics_folder, path_to_save):
    # load the model
    model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), alpha=1.0, pooling='avg', include_top=False)
    model.summary()

    img_dir = pics_folder

    with pics_data_file.open('r') as csvreader:
        data = csv.reader(csvreader, delimiter='\t')
        with path_to_save.open('w') as fw:
            for row in data:
                img_name = row[0] + '.jpg'
                img = img_dir / img_name
                try:
                    image = load_img(img, target_size=(224, 224))
                except Exception as e:
                    print(e)
                    continue

                image = img_to_array(image)
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                image = preprocess_input(image)
                feature = model.predict(image, verbose=0)[0]
                print(feature.shape)
                feature_str = (',').join([str(f) for f in feature])
                res = (',').join([row[0], feature_str])
                fw.write(res + '\n')


if __name__ == '__main__':
    print("Extract features")
    # extract features from all images
    # img_directory = Path('.\\Data\\images_mapping_train.csv')
    #     # file_save = Path('.\\google_train_features.csv')
    #     # pics_folder = Path('.\\Google_Train\\')
    #     #
    #     # extract_pic_features(img_directory, pics_folder, file_save)

    img_directory = Path('./images_mapping_test.csv')
    file_save = Path('./google_test_features.csv')
    pics_folder = Path('./Google_Test/')

    extract_pic_features(img_directory, pics_folder, file_save)

    # save to file
    # dump(features, open('.\\google_mobilenetv2_features.pkl', 'wb'))

