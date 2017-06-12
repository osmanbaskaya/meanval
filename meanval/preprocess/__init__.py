import collections
from itertools import cycle
import math
import numpy as np
import os
from constant import NUM_ALREADY_ALLOCATED_TOKEN, UNKNOWN_TOKEN_ID, END_OF_SENTENCE_ID
from data import Vocabulary

import tensorflow as tf
import nltk
import sys


def tokenize(out_fn, input_fn=None):
    with open(out_fn, 'wt') as out_file:
        if input_fn is None:
            input_f = sys.stdin
        else:
            input_f = open(input_fn)
        for line in input_f:
            line = nltk.word_tokenize(line)
            out_file.write(" ".join(line))
            out_file.write("\n")


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().split()


def _read_lines(filename):
    with tf.gfile.GFile(filename, "r") as f:
        sentences = f.read().splitlines()
        
    return sentences


def _read_labels(filename):
    labels = _read_lines(filename)
    labels =  np.array([int(label) for label in labels])
    return np.reshape(labels, (len(labels), 1))


def _build_vocab(filename):
    # TODO: add functionality to replace infrequent words with <unk>.
    data = _read_words(filename)
    vocabulary = Vocabulary.build(data, min_occurrence=1, num_already_allocated_tokens=NUM_ALREADY_ALLOCATED_TOKEN)
    return vocabulary


def _file_to_word_ids(filename, word_to_id):
    data = _read_lines(filename)
    sentences = []
    for sentence in data:
        word_ids = [word_to_id.get(word, UNKNOWN_TOKEN_ID) for word in sentence.split()]
        word_ids.append(END_OF_SENTENCE_ID)
        sentences.append(word_ids)

    return sentences


def read_data(word_to_id, data_path=None, dataset_type='train'):
    ref_fn = os.path.join(data_path, "wmt15.%s.ref.sentences.txt" % dataset_type)
    mt_fn = os.path.join(data_path, "wmt15.%s.mt.sentences.txt" % dataset_type)
    labels_fn = os.path.join(data_path, "wmt15.%s.labels.txt" % dataset_type)

    reference_sentences = _file_to_word_ids(ref_fn, word_to_id)
    mt_sentences = _file_to_word_ids(mt_fn, word_to_id)
    labels = _read_labels(labels_fn)

    return reference_sentences, mt_sentences, labels


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


def _prepare_batch_labels(labels, batch_size):

    num_of_batch = math.ceil((len(labels) / batch_size))
    label_batches = []
    for index in range(num_of_batch):
        label_batch = labels[index * batch_size:index*batch_size+batch_size]
        label_batches.append(label_batch)

    for label_batch in cycle(label_batches):
        yield label_batch


def _prepare_batch_sequence(sequences, batch_size):

    num_of_batch = math.ceil((len(sequences) / batch_size))
    batches = []
    length_batches = []
    for index in range(num_of_batch):
        # Batch related with Reference Translation
        batch = sequences[index * batch_size:index * batch_size + batch_size]

        # Lengths related with Reference Translation

        lengths = list(map(len, batch))
        max_seq_length = max(lengths)

        padded_batch = np.zeros(shape=[len(batch), max_seq_length], dtype=np.int32)
        for i, seq in enumerate(batch):
            for j, elem in enumerate(seq):
                padded_batch[i, j] = elem

        padded_batch = padded_batch.swapaxes(0, 1)
        length_batches.append(lengths)
        batches.append(padded_batch)

    for seq_batch, length_batch in zip(cycle(batches), cycle(length_batches)):
        yield seq_batch, length_batch


def transform_data(ref_sequences, mt_sequences, labels, batch_size):

    label_batch_it = _prepare_batch_labels(labels, batch_size)
    ref_seq_batch_it = _prepare_batch_sequence(ref_sequences, batch_size)
    mt_seq_batch_it = _prepare_batch_sequence(mt_sequences, batch_size)

    for (ref_seqs, ref_seq_lengths), (mt_seqs, mt_seq_lengths), labels, in zip(ref_seq_batch_it, mt_seq_batch_it,
                                                                               label_batch_it):
        yield ref_seqs, ref_seq_lengths, mt_seqs, mt_seq_lengths, labels


def prepare_data(dataset_type, data_path="datasets", batch_size=128):
    # FIXME: Use all the data to build vocabulary
    # Always use training data to build vocabulary.
    vocabulary = _build_vocab(os.path.join(data_path, "wmt15.train.ref.sentences.txt"))
    reference_sentences, mt_sentences, labels = read_data(vocabulary, data_path=data_path, dataset_type=dataset_type)
    return transform_data(reference_sentences, mt_sentences, labels, batch_size=batch_size), vocabulary


