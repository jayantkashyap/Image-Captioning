import numpy as np

from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers.wrappers import Bidirectional
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten

from model import nnmodel, data_generator

word2idx, idx2word, images, max_len, vocab_size, samples_per_epoch, encoding_train, encoding_test = nnmodel()

def predict_captions(image, final_model):
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        e = encoding_test[image[len(images):]]
        preds = final_model.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)

        if word_pred == "<end>" or len(start_word) > max_len:
            break

    return ' '.join(start_word[1:-1])


def beam_search_predictions(image, final_model, beam_index=3):
    start = [word2idx["<start>"]]

    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            e = encoding_test[image[len(images):]]
            preds = final_model.predict([np.array([e]), np.array(par_caps)])

            word_preds = np.argsort(preds[0])[-beam_index:]

            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []

    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption


def traininit():
    embedding_size = 300
    image_model = Sequential([
        Dense(embedding_size, input_shape=(2048,), activation='relu'),
        RepeatVector(max_len)
    ])

    caption_model = Sequential([
        Embedding(vocab_size, embedding_size, input_length=max_len),
        LSTM(256, return_sequences=True),
        TimeDistributed(Dense(300))
    ])

    final_model = Sequential([
        Merge([image_model, caption_model], mode='concat', concat_axis=1),
        Bidirectional(LSTM(256, return_sequences=False)),
        Dense(vocab_size),
        Activation('softmax')
    ])

    final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    final_model.summary()


    return final_model


def train():
    final_model = traininit()
    final_model.fit_generator(data_generator(encoding_train, vocab_size, max_len, batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, verbose=1)
    final_model.save_weights('time_inceptionV3_3.15_loss.h5')
    final_model.save_weights('time_inceptionV3_3.21_loss.h5')
    final_model.save_weights('time_inceptionV3_7_loss_3.2604.h5')

    return final_model


def predict(img, trn=False):
    if trn==True:
        final_model = train()
        final_model.load_weights('time_inceptionV3_3.15_loss.h5')
        final_model.load_weights('time_inceptionV3_3.21_loss.h5')
        final_model.load_weights('time_inceptionV3_7_loss_3.2604.h5')
        print('Normal Max search:',
              predict_captions(img, final_model))
        print('Beam Search, k=3:',
              beam_search_predictions(img, final_model, beam_index=3))
        print('Beam Search, k=5:',
              beam_search_predictions(img, final_model, beam_index=5))
        print('Beam Search, k=7:',
              beam_search_predictions(img, final_model, beam_index=7))
    else:
        final_model = traininit()
        final_model.load_weights('time_inceptionV3_3.15_loss.h5')
        final_model.load_weights('time_inceptionV3_3.21_loss.h5')
        final_model.load_weights('time_inceptionV3_7_loss_3.2604.h5')
        print('Normal Max search:',
              predict_captions(img, final_model))
        print('Beam Search, k=3:',
              beam_search_predictions(img, final_model, beam_index=3))
        print('Beam Search, k=5:',
              beam_search_predictions(img, final_model, beam_index=5))
        print('Beam Search, k=7:',
              beam_search_predictions(img, final_model, beam_index=7))