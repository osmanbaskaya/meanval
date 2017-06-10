from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from itertools import cycle
import math
import numpy as np
import os

import tensorflow as tf

UNKNOWN_TOKEN_ID = 0
END_OF_SENTENCE_ID = 1
PADDING_TOKEN_ID = 2

NUM_ALREADY_ALLOCATED_TOKEN = 3


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().split()


def _read_lines(filename):
    with tf.gfile.GFile(filename, "r") as f:
        sentences = f.read().splitlines()
        
    return sentences


def _read_labels(filename):
    labels = _read_lines(filename)
    return [int(label) for label in labels]


def _build_vocab(filename):
    # TODO: add functionality to replace infrequent words with <unk>.
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(NUM_ALREADY_ALLOCATED_TOKEN, len(words) + NUM_ALREADY_ALLOCATED_TOKEN)))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_lines(filename)
    sentences = []
    for sentence in data:
        word_ids = [word_to_id.get(word, UNKNOWN_TOKEN_ID) for word in sentence.split()]
        word_ids.append(END_OF_SENTENCE_ID)
        sentences.append(word_ids)

    return sentences


def read_data(word_to_id, data_path=None, dataset_type='train'):
    sentences_fn = os.path.join(data_path, "%s.sentences.txt" % dataset_type)
    labels_fn = os.path.join(data_path, "%s.labels.txt" % dataset_type)

    sentences = _file_to_word_ids(sentences_fn, word_to_id)
    labels = _read_labels(labels_fn)

    return sentences, labels


def data_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.
  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.
  Args:
    raw_data: one of the raw data outputs from evaluation_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).
  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
    with tf.name_scope(name, "DataProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y


def transform_data(sequences, labels, batch_size):

    num_of_batch = math.ceil((len(sequences) / batch_size))
    batches = []
    label_batches = []
    length_batches = []
    for index in range(num_of_batch):
        batch = sequences[index * batch_size:index*batch_size+batch_size]
        label_batch = labels[index * batch_size:index*batch_size+batch_size]
        label_batches.append(label_batch)

        lengths = list(map(len, batch))
        length_batches.append(lengths)
        max_seq_length = max(lengths)

        padded_batch = np.zeros(shape=[len(batch), max_seq_length], dtype=np.int32)
        for i, seq in enumerate(batch):
            for j, elem in enumerate(seq):
                padded_batch[i, j] = elem
        padded_batch = padded_batch.swapaxes(0, 1)
        batches.append(padded_batch)

    for seqs, labels, seq_lengths in zip(cycle(batches), cycle(label_batches), cycle(length_batches)):
        yield seqs, labels, seq_lengths


def prepare_data(dataset_type, data_path="", batch_size=128):
    # Always use training data to build vocabulary.
    word_to_id = _build_vocab(os.path.join(data_path, "train.sentences.txt"))
    sentences, labels = read_data(word_to_id, data_path=data_path, dataset_type=dataset_type)
    return transform_data(sentences, labels, batch_size=batch_size), len(word_to_id) + NUM_ALREADY_ALLOCATED_TOKEN


