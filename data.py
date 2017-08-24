import tensorflow as tf
import numpy as np
import sys, os

def load_dictionary():
    path = "ko_en_parallel_corpus"
    vocab_en_name = "vocab.en"

    with open(os.path.join(path, vocab_en_name)) as en:
        vocab_en = [line.rstrip() for line in en]
        en.close()

    en_dict = dict()
    en_dict['PAD'] = 0
    en_dict['SOS'] = 1

    for i, word in enumerate(vocab_en):
        en_dict[word] = i+2
    reverse_en_dict = {v: k for k, v in en_dict.iteritems()}

    return en_dict, reverse_en_dict

def load_data(dict, file, length=30):
    path = "ko_en_parallel_corpus"
    with open(os.path.join(path, file)) as f:
        sentences = [line.rstrip() for line in f]
        f.close()

    data_words = [sentence.split() for sentence in sentences]

    data = []

    for i, data_word in enumerate(data_words):
        datum = [dict[word] for word in data_word]

        if len(datum) < length:
            data.append(datum)

    X = np.zeros([len(data), length], np.int32)
    sequence_length = list()

    for i, x in enumerate(data):
        X[i] = np.lib.pad(x, [0, length - len(x)], 'constant', constant_values=(0, 0))
        sequence_length.append(len(x))

    return X, sequence_length, dict


def get_batch(batch_size=32, is_training=True):

    en_dict, reverse_dict = load_dictionary()

    if is_training:
        filename = "dict.en.train"
    else: filename = "dict.en.valid"

    X, seq_len, dict_ = load_data(en_dict, filename)
    data_len = len(X)
    num_batch = data_len // batch_size

    # X = tf.convert_to_tensor(X, tf.int32)
    # seq_len = tf.convert_to_tensor(seq_len, tf.int32)
    # input_queues = tf.train.slice_input_producer([X, seq_len])
    #
    # source, sequence_length = tf.train.shuffle_batch(input_queues,
    #                                 num_threads=8,
    #                                 batch_size=batch_size,
    #                                 capacity=batch_size * 64,
    #                                 min_after_dequeue=batch_size * 32,
    #                                 allow_smaller_final_batch=False)

    print "data loaded. (total data=%d, total batch=%d)" % (data_len, num_batch)

    return X, seq_len, dict_