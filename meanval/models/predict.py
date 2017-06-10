import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import numpy as np


class SimpleEncoderModel(object):

    def __init__(self):
        self.train_op = None
        self.loss = None
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
        self.decoder_targets = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_targets')

    def create_model(self, vocab_size):
        NUM_OF_RATINGS = 5
        vocab_size = vocab_size
        print(vocab_size)
        input_embedding_size = 20

        encoder_hidden_units = 20

        embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)
        encoder_cell = LSTMCell(encoder_hidden_units)

        ((encoder_fw_outputs,
          encoder_bw_outputs),
         (encoder_fw_final_state,
          encoder_bw_final_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                            cell_bw=encoder_cell,
                                            inputs=encoder_inputs_embedded,
                                            sequence_length=self.encoder_inputs_length,
                                            dtype=tf.float32, time_major=True))

        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        encoder_final_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )

        W = tf.Variable(tf.zeros([encoder_hidden_units * 2, NUM_OF_RATINGS]))
        b = tf.Variable(tf.zeros([NUM_OF_RATINGS]))

        # Construct model
        self.pred = tf.nn.softmax(tf.matmul(encoder_final_state.c, W) + b) # Softmax

        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=NUM_OF_RATINGS, dtype=tf.float32), logits=self.pred)

        loss = tf.reduce_mean(stepwise_cross_entropy)
        train_op = tf.train.AdamOptimizer().minimize(loss)

        self.train_op = train_op
        self.loss = loss

    def run(self, batch_iterator):

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(20):
            seqs, labels, seq_lengths = next(batch_iterator)
            feed_dict = {self.encoder_inputs: seqs,
                         self.encoder_inputs_length: seq_lengths,
                         self.decoder_targets: labels}

            _, l, predictions = sess.run([self.train_op, self.loss, self.pred], feed_dict=feed_dict)
            # print(np.argmax(predictions, axis=1))
            # print(predictions.shape)
            if i % 10 == 0:
                print('batch {}'.format(i))
                print('  minibatch loss: {}'.format(sess.run(self.loss, feed_dict)))
                predict_ = sess.run(self.pred, feed_dict)
                print(np.argmax(predict_, axis=1))
                # for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                #     print('  sample {}:'.format(i + 1))
                #     print('    input     > {}'.format(inp))
                #     print('    predicted > {}'.format(pred))
                #     if i >= 2:
                #         break
                # print()
