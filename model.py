from ops import *
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
from data import get_batch

__author__ = "soobinseo"

class ARAE(object):

    def __init__(self, LATENT_DIM=100, LEARNING_RATE=0.0005, LEARNING_RATE_CRITIC=0.00005, EPOCH=100000, BATCH_SIZE=32):
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
        self.LATENT_DIM = LATENT_DIM
        self.LEARNING_RATE = LEARNING_RATE
        self.LEARNING_RATE_CRITIC = LEARNING_RATE_CRITIC
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
        self.build_graph()

    def encoder(self, tensor, output_dim, is_mnist=True):
        """
        encode discrete feature to continuous latent vector
        :param tensor: 2-D tensor.
        :param output_dim: Integer. dimension of output
        :param is_mnist: Boolean. mnist or not
        :return: 2-D tensor. encoded latent vector
        """
        with tf.variable_scope("encoder"):

            fc_784 = fully_connected(tensor, 784, self.initializer, scope="fc_784")
            fc_800 = fully_connected(fc_784, 800, self.initializer, scope="fc_800")
            fc_400 = fully_connected(fc_800, 400, self.initializer, scope="fc_400")
            output = fully_connected(fc_400, output_dim, self.initializer, is_last=True, scope="encoder_output")

            return output

    def decoder(self, tensor, output_dim, is_mnist=True, reuse=False):
        """
        decode continuous vector to probability of pixel
        :param tensor: 2-D tensor.
        :param output_dim: Integer. dimension of output
        :param is_mnist: Boolean. mnist or not
        :param reuse: Boolean. reuse or not
        :return: 2-D tensor. decoded vector (image)
        """
        with tf.variable_scope("decoder", reuse=reuse):

            # add gausian noise
            tensor = gaussian_noise_layer(tensor, 0.4)
            fc_400 = fully_connected(tensor, 400, self.initializer, is_decoder=True, scope="fc_400")
            fc_800 = fully_connected(fc_400, 800, self.initializer, is_decoder=True, scope="fc_800")
            fc_1000 = fully_connected(fc_800, 1000, self.initializer, is_decoder=True, scope="fc_1000")
            output = fully_connected(fc_1000, output_dim, self.initializer, is_decoder=True, is_last=True, scope="decoder_output")


            return output

    def autoencoder(self, data):
        """
        deep autoencoder. reconstruction the input data
        :param data: 2-D tensor. data for reconstruction
        :return: 2-D tensor. reconstructed data and latent vector
        """
        with tf.variable_scope("autoencoder"):
            latent = self.encoder(data, self.LATENT_DIM)
            output = self.decoder(latent, data.get_shape()[-1].value)

            return output, latent

    def generator(self, z, reuse=False):
        """
        generator of WGAN
        :param z: 2-D tensor. noise with standard normal distribution
        :param reuse: Boolean. reuse or not
        :return: 2-D tensor. latent vector
        """
        with tf.variable_scope("generator", reuse=reuse):
            fc_64 = fully_connected(z, 64, initializer=self.initializer, scope="fc_64")
            fc_100 = fully_connected(fc_64, 100, initializer=self.initializer, scope="fc_100")
            fc_150 = fully_connected(fc_100, 150, initializer=self.initializer, scope="fc_150")
            latent = fully_connected(fc_150, self.LATENT_DIM, initializer=self.initializer, is_last=True, scope="generator_output")

        return latent

    def critic(self, latent, reuse=False):
        """
        discriminator of WGAN
        :param latent: 2-D tensor. latent vector
        :param reuse: Boolean. reuse or not
        :return: 2-D tensor. logit of data or noise
        """
        with tf.variable_scope("critic", reuse=reuse):

            fc_100 = fully_connected(latent, 100, initializer=self.initializer, scope="fc_100")
            fc_60 = fully_connected(fc_100, 60, initializer=self.initializer, scope="fc_60")
            fc_20 = fully_connected(fc_60, 20, initializer=self.initializer, scope="fc_20")
            output = fully_connected(fc_20, 1, initializer=self.initializer, is_last=True, scope="critic_output")

            # For using WGAN loss, do not activate
            # output = tf.nn.sigmoid(output)

        return output


    def build_graph(self):
        """
        build network
        :return:
        """
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.z = tf.placeholder(tf.float32, [None, 32], name="z")

        AE_out, real_latent = self.autoencoder(self.x)

        g_latent = self.generator(self.z)
        critic_real = self.critic(real_latent)
        critic_fake = self.critic(g_latent, reuse=True)

        # WGAN loss
        self.disc_real_loss = tf.reduce_mean(critic_real)
        self.disc_fake_loss = -tf.reduce_mean(critic_fake)
        self.critic_loss = tf.reduce_mean(critic_real-critic_fake)
        self.gen_loss = tf.reduce_mean(critic_fake)

        # for continous input, use L2norm
        self.AE_loss = tf.reduce_mean(tf.squared_difference(AE_out, self.x))

        # for discrete input, use cross entropy loss
        # self.AE_loss = tf.reduce_mean(self.x * tf.log(1 - AE_out) + (1-self.x) * tf.log(AE_out))

        # get trainable variables
        gen_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic")
        AE_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="autoencoder")
        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="autoencoder/encoder")

        # set optimizer for each module
        disc_op = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE_CRITIC)
        gen_op = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
        AE_op= tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)

        # compute gradients
        pos_critic_grad = disc_op.compute_gradients(self.disc_real_loss, critic_variables + encoder_variables)
        neg_critic_grad = disc_op.compute_gradients(self.disc_fake_loss, critic_variables)
        # clipping gradients for negative samples
        neg_critic_grad = [(tf.clip_by_value(grad, -0.05, 0.05), var) for grad, var in neg_critic_grad]
        gen_grad = gen_op.compute_gradients(self.gen_loss, gen_variables)
        AE_grad = AE_op.compute_gradients(self.AE_loss, AE_variables)

        # apply gradients
        self.update_critic_pos = disc_op.apply_gradients(pos_critic_grad)
        self.update_critic_neg = disc_op.apply_gradients(neg_critic_grad)
        self.update_G = gen_op.apply_gradients(gen_grad)
        self.update_AE = AE_op.apply_gradients(AE_grad)

        # reconstruction
        with tf.variable_scope("autoencoder"):
            self.real_pred = self.decoder(real_latent, 784, reuse=True)
            self.fake_pred = self.decoder(g_latent, 784, reuse=True)

    def train(self):
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            batch = self.mnist.train.next_batch(self.BATCH_SIZE)
            for i in range(self.EPOCH):
                # standard normal distribution for input noise z
                z_in = np.random.standard_normal(size=[self.BATCH_SIZE, 32]).astype(np.float32)

                # update AE
                _, _AEloss = sess.run([self.update_AE, self.AE_loss], feed_dict={self.x:batch[0]})

                # update critic & encoder at positive sample phase
                for j in range(10):
                    _, _critic_loss, real_loss = sess.run([self.update_critic_pos, self.critic_loss, self.disc_real_loss], feed_dict={self.x:batch[0], self.z:z_in})
                    _ = sess.run(self.update_critic_neg, feed_dict={self.x:batch[0], self.z:z_in})

                # update generator
                _, GAN_loss = sess.run([self.update_G, self.gen_loss], feed_dict={self.x:batch[0], self.z:z_in})

                if i % 10 == 0:
                    print "step %d, AEloss: %.4f,   real_loss: %.6f, gen_loss: %.6f, critic_loss:%.6f" %(i, _AEloss, real_loss, GAN_loss, _critic_loss)

                # save generated image
                if i % 1000 == 0:
                    img_real = sess.run(self.real_pred, feed_dict={self.x:batch[0]})
                    img_fake = sess.run(self.fake_pred, feed_dict={self.x:batch[0], self.z:z_in})
                    real_reshape = np.reshape(img_real, [-1, 28, 28])
                    fake_reshape = np.reshape(img_fake, [-1, 28, 28])
                    if not os.path.exists('./result'):
                        os.makedirs('./result')
                    plt.imsave("./result/real_%d.png"%i, real_reshape[0], cmap="gray")
                    plt.imsave("./result/fake_%d.png" % i, fake_reshape[0], cmap="gray")
