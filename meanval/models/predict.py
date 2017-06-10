import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import numpy as np


class SimpleEncoderModel(object):

    def __init__(self):
        self.train_op = None
        self.loss = None
        self.encoder_inputs_reference = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_ref_inputs')
        self.encoder_inputs_translation = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_mt_inputs')
        self.encoder_ref_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_ref_inputs_length')
        self.encoder_mt_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_mt_inputs_length')
        self.decoder_targets = tf.placeholder(shape=(None, 1), dtype=tf.int32, name='decoder_targets')
        self.pred = None

    def create_model(self, vocab_size):
        NUM_OF_RATINGS = 1
        vocab_size = vocab_size
        print("Vocabulary size: %s" % vocab_size)

        input_embedding_size = 20
        encoder_hidden_units = 20

        embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
        encoder_ref_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs_reference)
        encoder_mt_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs_translation)
        encoder_cell = LSTMCell(encoder_hidden_units)

        ((encoder_fw_outputs,
          encoder_bw_outputs),
         (encoder_fw_final_state,
          encoder_bw_final_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                            cell_bw=encoder_cell,
                                            inputs=encoder_ref_inputs_embedded,
                                            sequence_length=self.encoder_ref_inputs_length,
                                            dtype=tf.float32, time_major=True))
        ((encoder_fw_outputs2,
          encoder_bw_outputs2),
         (encoder_fw_final_state2,
          encoder_bw_final_state2)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                            cell_bw=encoder_cell,
                                            inputs=encoder_mt_inputs_embedded,
                                            sequence_length=self.encoder_mt_inputs_length,
                                            dtype=tf.float32, time_major=True))

        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c,
             encoder_fw_final_state2.c, encoder_bw_final_state2.c), 1)

        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        encoder_final_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )

        W = tf.Variable(tf.zeros([encoder_hidden_units * 4, NUM_OF_RATINGS]))
        b = tf.Variable(tf.zeros([NUM_OF_RATINGS]))

        # self.pred = tf.nn.softmax(tf.matmul(encoder_final_state.c, W) + b)  # Softmax

        # stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        #     labels=tf.one_hot(self.decoder_targets, depth=NUM_OF_RATINGS, dtype=tf.float32), logits=self.pred)
        # loss = tf.reduce_mean(stepwise_cross_entropy)

        self.pred = tf.matmul(encoder_final_state.c, W) + b
        loss = tf.losses.mean_squared_error(predictions=self.pred, labels=self.decoder_targets)

        train_op = tf.train.AdamOptimizer().minimize(loss)

        self.train_op = train_op
        self.loss = loss

    def get_feed_dict_for_next_batch(self, data_iterator):
        ref_seqs, ref_seq_length, mt_seqs, mt_seq_lengths, labels = next(data_iterator)
        return {self.encoder_inputs_reference: ref_seqs, self.encoder_inputs_translation: mt_seqs,
                self.encoder_ref_inputs_length: ref_seq_length,
                self.encoder_mt_inputs_length: mt_seq_lengths, self.decoder_targets: labels}

    def run(self, training_batch_iterator, validation_batch_iterator):

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            feed_dict = self.get_feed_dict_for_next_batch(training_batch_iterator)
            _, training_loss, predictions = sess.run([self.train_op, self.loss, self.pred], feed_dict=feed_dict)
            if i % 1000 == 0:
                feed_dict = self.get_feed_dict_for_next_batch(validation_batch_iterator)
                validation_loss = sess.run([self.loss], feed_dict=feed_dict)
                print('Batch {}'.format(i))
                print('\tMini-batch Training loss: {}'.format(training_loss))
                print('\tMini-batch Validation loss: {}'.format(validation_loss))
