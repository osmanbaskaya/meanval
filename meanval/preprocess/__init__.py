from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf

END_OF_SENTENCE = "<eos>"
UNKNOWN_TOKEN_ID = 0


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("\n", END_OF_SENTENCE).split()


def _read_sentences(filename):
    with tf.gfile.GFile(filename, "r") as f:
        sentences = f.read().decode("utf-8").splitlines()
        
    for sentence in sentences:
        sentence.append(END_OF_SENTENCE)

    return sentences


def _build_vocab(filename):
    # TODO: add functionality to replace infrequent words with <unk>.
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(1, len(words) + 1)))  # 0 is UNKNOWN_TOKEN

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_sentences(filename)
    sentences = []
    for sentence in data:
        sentences.append([word_to_id.get(word, UNKNOWN_TOKEN_ID) for word in sentence])

    return sentences


def evaluation_raw_data(data_path=None):
    # TODO: Train / Validate / Test will be edited.
    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "train.txt")
    test_path = os.path.join(data_path, "train.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary_size = len(word_to_id)

    return train_data, valid_data, test_data, vocabulary_size


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
