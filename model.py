import glob
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm

from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing import sequence
from keras.applications.inception_v3 import InceptionV3



model = InceptionV3(weights='imagenet')
token = 'Image-Captioning/Flickr8k/Flickr8k.token.txt'
images = 'Image-Captioning/Flickr8k/Flicker8k_Dataset/'
train_images_file = 'Image-Captioning/Flickr8k/Flickr_8k.trainImages.txt'
val_images_file = 'Image-Captioning/Flickr8k/Flickr_8k.devImages.txt'
test_images_file = 'Image-Captioning/Flickr8k/Flickr_8k.testImages.txt'
word2idx = {}
idx2word = {}

img = glob.glob(images+'*.jpg')
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))
val_images = set(open(val_images_file, 'r').read().strip().split('\n'))
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

new_input = model.input
hidden_layer = model.layers[-2].output

model_new = Model(new_input, hidden_layer)


def pre_model():
    new_input = model.input
    hidden_layer = model.layers[-2].output
    model_new = Model(new_input, hidden_layer)
    return model_new


def caption_dict(captions):
    d = {}
    for i, row in enumerate(captions):
        row = row.split('\t')
        row[0] = row[0][:len(row[0]) - 2]
        if row[0] in d:
            d[row[0]].append(row[1])
        else:
            d[row[0]] = [row[1]]
    return d


def split_data(l):
    temp = []
    for i in img:
        if i[len(images):] in l:
            temp.append(i)
    return temp


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x


def encode(image):
    image = preprocess(image)
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc


def write_captions_to_file(train_d):
    f = open('Image-Captioning/flickr8k_training_dataset.txt', 'w')
    f.write("image_id\tcaptions\n")
    for key, val in train_d.items():
        for i in val:
            f.write(key[len(images):] + "\t" + "<start> " + i + " <end>" + "\n")

    f.close()


def data_generator(encoding_train, vocab_size, max_len, batch_size=32):
    partial_caps = []
    next_words = []
    images = []

    df = pd.read_csv('Image-Captioning/flickr8k_training_dataset.txt', delimiter='\t')
    df = df.sample(frac=1)
    iter = df.iterrows()
    c = []
    imgs = []
    for i in range(df.shape[0]):
        x = next(iter)
        c.append(x[1][1])
        imgs.append(x[1][0])

    count = 0
    while True:
        for j, text in enumerate(c):
            current_image = encoding_train[imgs[j]]
            for i in range(len(text.split()) - 1):
                count += 1

                partial = [word2idx[txt] for txt in text.split()[:i + 1]]
                partial_caps.append(partial)

                n = np.zeros(vocab_size)
                n[word2idx[text.split()[i + 1]]] = 1
                next_words.append(n)

                images.append(current_image)

                if count >= batch_size:
                    next_words = np.asarray(next_words)
                    images = np.asarray(images)
                    partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_len, padding='post')
                    yield [[images, partial_caps], next_words]
                    partial_caps = []
                    next_words = []
                    images = []
                    count = 0


def nnmodel():
    encoding_train = {}
    encoding_test = {}
    train_d = {}
    val_d = {}
    test_d = {}
    caps = []
    unique = []

    samples_per_epoch = 0

    captions = open(token, 'r').read().strip().split('\n')
    d = caption_dict(captions=captions)

    train_img = split_data(train_images)
    val_img = split_data(val_images)
    test_img = split_data(test_images)

    for img in tqdm(train_img):
        encoding_train[img[len(images):]] = encode(img)
    for img in tqdm(test_img):
        encoding_test[img[len(images):]] = encode(img)

    with open("Image-Captioning/encoded_images_inceptionV3.p", "wb") as encoded_pickle:
        pickle.dump(encoding_train, encoded_pickle)
    with open("Image-Captioning/encoded_images_test_inceptionV3.p", "wb") as encoded_pickle:
        pickle.dump(encoding_test, encoded_pickle)

    for i in train_img:
        if i[len(images):] in d:
            train_d[i] = d[i[len(images):]]
    for i in val_img:
        if i[len(images):] in d:
            val_d[i] = d[i[len(images):]]
    for i in test_img:
        if i[len(images):] in d:
            test_d[i] = d[i[len(images):]]

    for key, val in train_d.items():
        for i in val:
            caps.append('<start> ' + i + ' <end>')

    words = [i.split() for i in caps]

    for i in words:
        unique.extend(i)

    unique = list(set(unique))
    vocab_size = len(unique)
    word2idx = {val: index for index, val in enumerate(unique)}
    idx2word = {index: val for index, val in enumerate(unique)}

    max_len = 0
    for c in caps:
        c = c.split()
        if len(c) > max_len:
            max_len = len(c)

    df = pd.read_csv('Image-Captioning/flickr8k_training_dataset.txt', delimiter='\t')
    c = [i for i in df['captions']]
    imgs = [i for i in df['image_id']]

    for ca in caps:
        samples_per_epoch += len(ca.split()) - 1

    return word2idx, idx2word, images, max_len, vocab_size, samples_per_epoch, encoding_train, encoding_test

if __name__ == '__main__':
    nnmodel()