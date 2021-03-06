import tensorflow as tf
import sys
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple


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

    def create_model(self, vocabulary, encoder_hidden_unit_size=20, embedding_weights=None, embedding_size=None):

        print("Vocabulary size: %s" % vocabulary.size)

        input_embedding_size = embedding_size

        if embedding_weights is None:
            assert input_embedding_size is not None, "user should provide either embedding_size or embedding_weights"
            embeddings = tf.Variable(tf.random_uniform([vocabulary.size, input_embedding_size], -1.0, 1.0),
                                     dtype=tf.float32)
        else:
            print("Embedding weights is used.", file=sys.stderr)
            print("Embeddings shape {}".format(embedding_weights.shape), file=sys.stderr)
            embeddings = tf.Variable(embedding_weights, dtype=tf.float32, trainable=False)

        encoder_ref_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs_reference)
        encoder_mt_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs_translation)
        encoder_cell = LSTMCell(encoder_hidden_unit_size)

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

        W = tf.Variable(tf.zeros([encoder_hidden_unit_size * 4, 1]))
        b = tf.Variable(tf.zeros([1]))

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
            _, training_loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            if i % 1000 == 0:
                feed_dict = self.get_feed_dict_for_next_batch(validation_batch_iterator)
                validation_loss = sess.run([self.loss], feed_dict=feed_dict)
                print('Batch {}'.format(i))
                print('\tMini-batch Training loss: {}'.format(training_loss))
                print('\tMini-batch Validation loss: {}'.format(validation_loss))
