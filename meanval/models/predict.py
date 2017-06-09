import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import models
import models.helpers


class SimpleEncoderModel(object):

    def __init__(self):
        self.train_op = None
        self.loss = None
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
        self.decoder_targets = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_targets')

    def create_model(self):
        NUM_OF_RATINGS = 5

        vocab_size = 10
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

        #loss function
        loss = tf.reduce_mean(stepwise_cross_entropy)
        #train it
        train_op = tf.train.AdamOptimizer().minimize(loss)

        self.train_op = train_op
        self.loss = loss

    def run(self):
        PAD = 0
        EOS = 1

        batch_size = 100
        batches = models.helpers.random_sequences(length_from=3, length_to=8,
                                                  vocab_lower=2, vocab_upper=10,
                                                  batch_size=batch_size)
        def next_feed():
            batch = next(batches)
            encoder_inputs_, encoder_input_lengths_ = models.helpers.batch(batch)
            decoder_targets_, _ = models.helpers.batch(
                [(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
            )
            decoder_targets_ = np.random.randint(1, 6, size=batch_size)
            return {
                self.encoder_inputs: encoder_inputs_,
                self.encoder_inputs_length: encoder_input_lengths_,
                self.decoder_targets: decoder_targets_,
            }

        print('head of the batch:')
        for seq in next(batches)[:10]:
            print(seq)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        batches_in_epoch = 100

        for batch in range(5):
                fd = next_feed()
                _, l, predictions = sess.run([self.train_op, self.loss, self.pred], fd)
                # print(np.argmax(predictions, axis=1))
                # print(predictions.shape)
                if batch == 0 or batch % batches_in_epoch == 0:
                    print('batch {}'.format(batch))
                    print('  minibatch loss: {}'.format(sess.run(self.loss, fd)))
                    predict_ = sess.run(self.pred, fd)
                    # print(predict_)
                    # for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                    #     print('  sample {}:'.format(i + 1))
                    #     print('    input     > {}'.format(inp))
                    #     print('    predicted > {}'.format(pred))
                    #     if i >= 2:
                    #         break
                    # print()

model = SimpleEncoderModel()
model.create_model()
model.run()
