from ops import *
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
from data import get_batch

__author__ = "soobinseo"


class ARAE_text(object):
    def __init__(self, EPOCH=100000, batch_size=32, embedding_size=300,
                 num_units=300):
        """
        https://arxiv.org/abs/1706.04223
        While handling discrete outputs, gradients do not flow over the network parameters, so this paper
        demonstrates a method which mapping discrete feature to continuous latent space by AE and WGAN.

        :param LATENT_DIM: Integer. dimension of latent space
        :param LEARNING_RATE: Float. learning rate for optimizing generator and autoencoder
        :param LEARNING_RATE_CRITIC: Float. learning rate for optimizing critic
        :param EPOCH: Integer. # of epochs
        :param BATCH_SIZE: Integer. batch size
        """
        self.initializer = tf.truncated_normal_initializer(stddev=0.02)
        self.AE_learning_rate = 1.
        self.critic_lr= 0.00001
        self.gen_lr = 0.00005
        self.EPOCH = EPOCH
        self.batch_size = batch_size
        self.num_units = num_units
        self.data, self.sequence_length, self.dict = get_batch()
        self.num_batch = len(self.data) // self.batch_size
        self.reverse_dict = {v: k for k, v in self.dict.iteritems()}
        self.voca_size = len(self.dict)
        self.max_len = 30
        self.embedding_size = embedding_size
        with tf.variable_scope("embedding"):
            self.embedding = tf.get_variable("embedding_table", [self.voca_size, self.embedding_size])
        self.build_graph()

    def encoder(self, tensor):
        """
        encode discrete feature to continuous latent vector
        :param tensor: 2-D tensor.
        :param output_dim: Integer. dimension of output
        :param is_mnist: Boolean. mnist or not
        :return: 2-D tensor. encoded latent vector
        """
        with tf.variable_scope("encoder"):
            tensor = tf.nn.embedding_lookup(self.embedding, tensor)
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
            outputs, state = tf.nn.dynamic_rnn(cell, tensor, sequence_length=self.seq_len, dtype=tf.float32)
            output = outputs[:,-1,:]
            output = tf.nn.l2_normalize(output, -1)

            return output

    def decoder(self, tensor, reuse=False):
        """
        decode continuous vector to probability of pixel
        :param tensor: 2-D tensor.
        :param output_dim: Integer. dimension of output
        :param is_mnist: Boolean. mnist or not
        :param reuse: Boolean. reuse or not
        :return: 2-D tensor. decoded vector (image)
        """

        outputs, predictions = [], []

        with tf.variable_scope("decoder", reuse=reuse) as scope:


            # add gausian noise
            decoder_input = gaussian_noise_layer(tensor, 0.2)
            encoder_dim = tensor.get_shape().as_list()[-1]
            W = tf.get_variable("decoder_last_weight", [self.num_units + encoder_dim, self.voca_size])
            b = tf.get_variable("decoder_last_bias", [self.voca_size])
            # time-major: [batch_size, max_len, num_units] --> [max_len, batch_size, num_units]
            # decoder_input = tf.transpose(decoder_input, [1,0,2])
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units, state_is_tuple=False)
            # initial_state = state = decoder_input
            initial_state = tf.zeros([self.batch_size, self.num_units])
            initial_state = tf.concat([initial_state, decoder_input], 1)


            for i in range(self.max_len):
                if i == 0:
                    # start of sequence
                    input_ = tf.nn.embedding_lookup(self.embedding, tf.ones([self.batch_size], dtype=tf.int32))
                    state = initial_state

                else:
                    scope.reuse_variables()
                    input_ = tf.nn.embedding_lookup(self.embedding, prediction)

                output, state = cell(input_, state)
                output = tf.concat([output, tensor], -1)
                output = tf.nn.xw_plus_b(output, W, b)

                prediction = tf.argmax(output, axis=1)

                outputs.append(output)
                predictions.append(prediction)

            predictions = tf.transpose(tf.stack(predictions), [1,0])
            outputs = tf.stack(outputs)

            return predictions, outputs

    def autoencoder(self, data):
        """
        deep autoencoder. reconstruction the input data
        :param data: 2-D tensor. data for reconstruction
        :return: 2-D tensor. reconstructed data and latent vector
        """
        with tf.variable_scope("autoencoder"):
            latent = self.encoder(data)
            _, output = self.decoder(latent)

            return output, latent

    def generator(self, z, reuse=False):
        """
        generator of WGAN
        :param z: 2-D tensor. noise with standard normal distribution
        :param reuse: Boolean. reuse or not
        :return: 2-D tensor. latent vector
        """
        with tf.variable_scope("generator", reuse=reuse):
            fc_300 = fully_connected(z, 300, initializer=self.initializer, scope="fc_300")
            latent = fully_connected(fc_300, self.num_units, initializer=self.initializer, is_last=True,
                                     scope="generator_output")

        return latent

    def critic(self, latent, reuse=False):
        """
        discriminator of WGAN
        :param latent: 2-D tensor. latent vector
        :param reuse: Boolean. reuse or not
        :return: 2-D tensor. logit of data or noise
        """
        with tf.variable_scope("critic", reuse=reuse):
            fc_300 = fully_connected(latent, 300, initializer=self.initializer, scope="fc_300")
            output = fully_connected(fc_300, 1, initializer=self.initializer, is_last=True, scope="critic_output")

            # For using WGAN loss, do not activate
            # output = tf.nn.sigmoid(output)

        return output

    def build_graph(self):
        """
        build network
        :return:
        """
        self.x = tf.placeholder(tf.int32, [None, self.max_len], name="input_")
        self.z = tf.placeholder(tf.float32, [None, 100], name="z")
        self.seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
        labels = tf.one_hot(self.x, self.voca_size)

        AE_out, real_latent = self.autoencoder(self.x)

        g_latent = self.generator(self.z)
        critic_real = self.critic(real_latent)
        critic_fake = self.critic(g_latent, reuse=True)

        # WGAN loss
        self.disc_real_loss = tf.reduce_mean(critic_real)
        self.disc_fake_loss = -tf.reduce_mean(critic_fake)
        self.critic_loss = tf.reduce_mean(critic_real) - tf.reduce_mean(critic_fake)
        self.gen_loss = tf.reduce_mean(critic_fake)

        # for discrete input, use cross entropy loss
        self.AE_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=AE_out))

        # get trainable variables
        gen_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic")
        AE_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="autoencoder")
        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="autoencoder/encoder")

        # set optimizer for each module
        disc_op = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        gen_op = tf.train.AdamOptimizer(learning_rate=self.gen_lr)
        AE_op = tf.train.GradientDescentOptimizer(learning_rate=self.AE_learning_rate)

        # compute gradients
        pos_critic_grad = disc_op.compute_gradients(self.disc_real_loss, critic_variables + encoder_variables)
        neg_critic_grad = disc_op.compute_gradients(self.disc_fake_loss, critic_variables)
        # clipping gradients for negative samples
        neg_critic_grad = [(tf.clip_by_value(grad, -0.01, 0.01), var) for grad, var in neg_critic_grad]
        gen_grad = gen_op.compute_gradients(self.gen_loss, gen_variables)
        AE_grad = AE_op.compute_gradients(self.AE_loss, AE_variables)
        AE_grad = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in AE_grad]

        # apply gradients
        self.update_critic_pos = disc_op.apply_gradients(pos_critic_grad)
        self.update_critic_neg = disc_op.apply_gradients(neg_critic_grad)
        self.update_G = gen_op.apply_gradients(gen_grad)
        self.update_AE = AE_op.apply_gradients(AE_grad)

        # reconstruction
        with tf.variable_scope("autoencoder"):
            self.real_pred, _ = self.decoder(real_latent, reuse=True)
            self.fake_pred, _ = self.decoder(g_latent, reuse=True)

    def train(self):
        init = tf.global_variables_initializer()


        with tf.Session() as sess:
            sess.run(init)
            # batch = self.mnist.train.next_batch(self.BATCH_SIZE)
            for i in range(self.EPOCH):
                for j in range(self.num_batch):
                    x = self.data[j*self.batch_size:(j+1)*self.batch_size]
                    seq_len = self.sequence_length[j*self.batch_size:(j+1)*self.batch_size]
                    # standard normal distribution for input noise z
                    z_in = np.random.standard_normal(size=[self.batch_size, 100]).astype(np.float32)

                    # update AE
                    _, _AEloss = sess.run([self.update_AE, self.AE_loss], feed_dict={self.x:x, self.seq_len:seq_len})

                    # update critic & encoder at positive sample phase
                    for k in range(5):
                        _, _critic_loss, real_loss = sess.run(
                            [self.update_critic_pos, self.critic_loss, self.disc_real_loss],
                            feed_dict={self.x:x, self.seq_len:seq_len, self.z: z_in})
                        _ = sess.run(self.update_critic_neg, feed_dict={self.z: z_in})

                    # update generator
                    _, GAN_loss = sess.run([self.update_G, self.gen_loss], feed_dict={self.z: z_in})

                    if j % 10 == 0:
                        print "step %d, AEloss: %.4f,   real_loss: %.6f, gen_loss: %.6f, critic_loss:%.6f" % (
                        i, _AEloss, real_loss, GAN_loss, _critic_loss)

                        real_pred = sess.run(self.real_pred, feed_dict={self.x:x, self.seq_len:seq_len})
                        fake_pred = sess.run(self.fake_pred, feed_dict={self.z:z_in})

                        print [self.reverse_dict[idx] for idx in x[0]]
                        print [self.reverse_dict[idx] for idx in real_pred[0]]
                        print [self.reverse_dict[idx] for idx in fake_pred[0]]


                # save generated image

