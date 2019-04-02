import numpy as np
import timeit
import tensorflow as tf
import os

# npz file 만드는 법

# DEFINE
TRAIN = True
TRAIN_DATA_PATH = 'eye_tracker_train_and_val.npz'
LEARNING_RATE = 0.0025
BATCH_SIZE = 100
MAX_EPOCH = 100
PATIENCE = 5
PRINT_PER_EPOCH = 1

SAVE_MODEL = 'my_model'
SAVE_LOSS = 'loss.npz'

# PARAMETERS
img_size = 64
n_channel = 3
mask_size = 25

conv1_eye_size = 11
conv1_eye_out = 96
pool1_eye_size = 2
pool1_eye_stride = 2

conv2_eye_size = 5
conv2_eye_out = 256
pool2_eye_size = 2
pool2_eye_stride = 2

conv3_eye_size = 3
conv3_eye_out = 384
pool3_eye_size = 2
pool3_eye_stride = 2

conv4_eye_size = 1
conv4_eye_out = 64
pool4_eye_size = 2
pool4_eye_stride = 2

eye_size = 2 * 2 * 2 * conv4_eye_out

conv1_face_size = 11
conv1_face_out = 96
pool1_face_size = 2
pool1_face_stride = 2

conv2_face_size = 5
conv2_face_out = 256
pool2_face_size = 2
pool2_face_stride = 2

conv3_face_size = 3
conv3_face_out = 384
pool3_face_size = 2
pool3_face_stride = 2

conv4_face_size = 1
conv4_face_out = 64
pool4_face_size = 2
pool4_face_stride = 2

face_size = 2 * 2 * conv4_face_out

fc_eye_size = 128
fc_face_size = 128
fc_face_mask_size = 256
face_face_mask_size = 128
fc_size = 128
fc2_size = 2


def load_data(file):
    npzfile = np.load(file)

    train_eye_left = npzfile["train_eye_left"]
    train_eye_right = npzfile["train_eye_right"]
    train_face = npzfile["train_face"]
    train_face_mask = npzfile["train_face_mask"]
    train_y = npzfile["train_y"]

    val_eye_left = npzfile["val_eye_left"]
    val_eye_right = npzfile["val_eye_right"]
    val_face = npzfile["val_face"]
    val_face_mask = npzfile["val_face_mask"]
    val_y = npzfile["val_y"]

    return [train_eye_left, train_eye_right, train_face, train_face_mask, train_y], [val_eye_left, val_eye_right,
                                                                                     val_face, val_face_mask, val_y]


def normalize(data):
    shape = data.shape
    data = np.reshape(data, (shape[0], -1))
    data = data.astype('float32') / 255.
    data = data - np.mean(data, axis=0)

    return np.reshape(data, shape)


def prepare_data(data):
    eye_left, eye_right, face, face_mask, y = data

    print("BEFORE")
    print(eye_left.shape, face.shape, face_mask.shape)

    eye_left = normalize(eye_left)
    eye_right = normalize(eye_right)
    face = normalize(face)
    face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
    y = y.astype('float32')

    print("AFTER")
    print(eye_left.shape, face.shape, face_mask.shape)

    return [eye_left, eye_right, face, face_mask, y]


def shuffle_data(data):
    idx = np.arange(data[0].shape[0])
    np.random.shuffle(idx)

    for i in range(len(data)):
        data[i] = data[i][idx]

    return data


def next_batch(data, batch_size):
    for i in np.arange(0, data[0].shape[0], batch_size):
        yield [each[i: i + batch_size] for each in data]


class EyeTracker(object):

    def __init__(self):

        self.eye_left = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='eye_left')
        self.eye_right = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='eye_right')
        self.face = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='face')
        self.face_mask = tf.placeholder(tf.float32, [None, mask_size * mask_size], name='face_mask')

        self.y = tf.placeholder(tf.float32, [None, 2], name='pos')

        self.weights = {

            'conv1_eye': tf.get_variable('conv1_eye_w',
                                         shape=(conv1_eye_size, conv1_eye_size, n_channel, conv1_eye_out),
                                         initializer=tf.contrib.layers.xavier_initializer()),
            'conv2_eye': tf.get_variable('conv2_eye_w',
                                         shape=(conv2_eye_size, conv2_eye_size, conv1_eye_out, conv2_eye_out),
                                         initializer=tf.contrib.layers.xavier_initializer()),
            'conv3_eye': tf.get_variable('conv3_eye_w',
                                         shape=(conv3_eye_size, conv3_eye_size, conv2_eye_out, conv3_eye_out),
                                         initializer=tf.contrib.layers.xavier_initializer()),
            'conv4_eye': tf.get_variable('conv4_eye_w',
                                         shape=(conv4_eye_size, conv4_eye_size, conv3_eye_out, conv4_eye_out),
                                         initializer=tf.contrib.layers.xavier_initializer()),

            'conv1_face': tf.get_variable('conv1_face_w',
                                          shape=(conv1_face_size, conv1_face_size, n_channel, conv1_face_out),
                                          initializer=tf.contrib.layers.xavier_initializer()),
            'conv2_face': tf.get_variable('conv2_face_w',
                                          shape=(conv2_face_size, conv2_face_size, conv1_face_out, conv2_face_out),
                                          initializer=tf.contrib.layers.xavier_initializer()),
            'conv3_face': tf.get_variable('conv3_face_w',
                                          shape=(conv3_face_size, conv3_face_size, conv2_face_out, conv3_face_out),
                                          initializer=tf.contrib.layers.xavier_initializer()),
            'conv4_face': tf.get_variable('conv4_face_w',
                                          shape=(conv4_face_size, conv4_face_size, conv3_face_out, conv4_face_out),
                                          initializer=tf.contrib.layers.xavier_initializer()),

            'fc_eye': tf.get_variable('fc_eye_w',
                                      shape=(eye_size, fc_eye_size),
                                      initializer=tf.contrib.layers.xavier_initializer()),
            'fc_face': tf.get_variable('fc_face_w',
                                       shape=(face_size, fc_face_size),
                                       initializer=tf.contrib.layers.xavier_initializer()),
            'fc_face_mask': tf.get_variable('fc_face_mask_w',
                                            shape=(mask_size * mask_size, fc_face_mask_size),
                                            initializer=tf.contrib.layers.xavier_initializer()),

            'face_face_mask': tf.get_variable('face_face_mask_w',
                                              shape=(fc_face_size + fc_face_mask_size, face_face_mask_size),
                                              initializer=tf.contrib.layers.xavier_initializer()),

            'fc': tf.get_variable('fc_w',
                                  shape=(fc_eye_size + face_face_mask_size, fc_size),
                                  initializer=tf.contrib.layers.xavier_initializer()),
            'fc2': tf.get_variable('fc2_w',
                                   shape=(fc_size, fc2_size),
                                   initializer=tf.contrib.layers.xavier_initializer())

        }

        self.biases = {

            'conv1_eye': tf.Variable(tf.constant(0.1, shape=[conv1_eye_out])),
            'conv2_eye': tf.Variable(tf.constant(0.1, shape=[conv2_eye_out])),
            'conv3_eye': tf.Variable(tf.constant(0.1, shape=[conv3_eye_out])),
            'conv4_eye': tf.Variable(tf.constant(0.1, shape=[conv4_eye_out])),

            'conv1_face': tf.Variable(tf.constant(0.1, shape=[conv1_face_out])),
            'conv2_face': tf.Variable(tf.constant(0.1, shape=[conv2_face_out])),
            'conv3_face': tf.Variable(tf.constant(0.1, shape=[conv3_face_out])),
            'conv4_face': tf.Variable(tf.constant(0.1, shape=[conv4_face_out])),

            'fc_eye': tf.Variable(tf.constant(0.1, shape=[fc_eye_size])),
            'fc_face': tf.Variable(tf.constant(0.1, shape=[fc_face_size])),
            'fc_face_mask': tf.Variable(tf.constant(0.1, shape=[fc_face_mask_size])),
            'face_face_mask': tf.Variable(tf.constant(0.1, shape=[face_face_mask_size])),

            'fc': tf.Variable(tf.constant(0.1, shape=[fc_size])),
            'fc2': tf.Variable(tf.constant(0.1, shape=[fc2_size]))

        }

        self.pred = self.itracker_nets(self.eye_left, self.eye_right, self.face, self.face_mask, self.weights,
                                       self.biases)

    def conv2d(self, x, W, b, strides=1):

        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)

        return tf.nn.relu(x)

    def maxpool2d(self, x, k, strides):

        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1], padding='VALID')

    def itracker_nets(self, eye_left, eye_right, face, face_mask, weights, biases):

        eye_left = self.conv2d(eye_left, weights['conv1_eye'], biases['conv1_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool1_eye_size, strides=pool1_eye_stride)
        eye_left = self.conv2d(eye_left, weights['conv2_eye'], biases['conv2_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool2_eye_size, strides=pool2_eye_stride)
        eye_left = self.conv2d(eye_left, weights['conv3_eye'], biases['conv3_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool3_eye_size, strides=pool3_eye_stride)
        eye_left = self.conv2d(eye_left, weights['conv4_eye'], biases['conv4_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool4_eye_size, strides=pool4_eye_stride)

        eye_right = self.conv2d(eye_right, weights['conv1_eye'], biases['conv1_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool1_eye_size, strides=pool1_eye_stride)
        eye_right = self.conv2d(eye_right, weights['conv2_eye'], biases['conv2_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool2_eye_size, strides=pool2_eye_stride)
        eye_right = self.conv2d(eye_right, weights['conv3_eye'], biases['conv3_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool3_eye_size, strides=pool3_eye_stride)
        eye_right = self.conv2d(eye_right, weights['conv4_eye'], biases['conv4_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool4_eye_size, strides=pool4_eye_stride)

        face = self.conv2d(face, weights['conv1_face'], biases['conv1_face'], strides=1)
        face = self.maxpool2d(face, k=pool1_face_size, strides=pool1_face_stride)
        face = self.conv2d(face, weights['conv2_face'], biases['conv2_face'], strides=1)
        face = self.maxpool2d(face, k=pool2_face_size, strides=pool2_face_stride)
        face = self.conv2d(face, weights['conv3_face'], biases['conv3_face'], strides=1)
        face = self.maxpool2d(face, k=pool3_face_size, strides=pool3_face_stride)
        face = self.conv2d(face, weights['conv4_face'], biases['conv4_face'], strides=1)
        face = self.maxpool2d(face, k=pool4_face_size, strides=pool4_face_stride)

        eye_left = tf.reshape(eye_left, [-1, int(np.prod(eye_left.get_shape()[1:]))])
        eye_right = tf.reshape(eye_right, [-1, int(np.prod(eye_right.get_shape()[1:]))])

        eye = tf.concat([eye_left, eye_right], 1)
        eye = tf.nn.relu(tf.add(tf.matmul(eye, weights['fc_eye']), biases['fc_eye']))

        face = tf.reshape(face, [-1, int(np.prod(face.get_shape()[1:]))])
        face = tf.nn.relu(tf.add(tf.matmul(face, weights['fc_face']), biases['fc_face']))

        face_mask = tf.nn.relu(tf.add(tf.matmul(face_mask, weights['fc_face_mask']), biases['fc_face_mask']))

        face_face_mask = tf.concat([face, face_mask], 1)
        face_face_mask = tf.nn.relu(
            tf.add(tf.matmul(face_face_mask, weights['face_face_mask']), biases['face_face_mask']))

        fc = tf.concat([eye, face_face_mask], 1)
        fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['fc']), biases['fc']))

        out = tf.add(tf.matmul(fc, weights['fc2']), biases['fc2'], name="output")

        return out

    def train(self, train_data, val_data, lr=1e-3, batch_size=100, max_epoch=100, min_delta=1e-4,
              patience=5, print_per_epoch=1, out_model='my_model'):

        ckpt = os.path.split(out_model)[0]
        if not os.path.exists(ckpt):
            os.mkdir(ckpt)
        print()

        self.cost = tf.losses.mean_squared_error(self.y, self.pred)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)

        self.err = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(self.pred, self.y), axis=1)))

        train_loss_history = []
        train_err_history = []

        val_loss_history = []
        val_err_history = []

        n_incr_error = 0
        best_loss = np.Inf

        n_batches = train_data[0].shape[0] / batch_size + (train_data[0].shape[0] % batch_size != 0)
        val_n_batches = val_data[0].shape[0] / batch_size + (val_data[0].shape[0] % batch_size != 0)

        tf.get_collection("validation_nodes")

        tf.add_to_collection("validation_nodes", self.eye_left)
        tf.add_to_collection("validation_nodes", self.eye_right)
        tf.add_to_collection("validation_nodes", self.face)
        tf.add_to_collection("validation_nodes", self.face_mask)
        tf.add_to_collection("validation_nodes", self.pred)

        saver = tf.train.Saver(max_to_keep=1)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            writer = tf.summary.FileWriter("logs", sess.graph)

            for n_epoch in range(1, max_epoch + 1):

                n_incr_error += 1
                train_loss = 0.
                train_err = 0.
                train_data = shuffle_data(train_data)

                for batch_train_data in next_batch(train_data, batch_size):
                    ####################
                    sess.run(self.optimizer, feed_dict={self.eye_left: batch_train_data[0],
                                                        self.eye_right: batch_train_data[1],
                                                        self.face: batch_train_data[2],
                                                        self.face_mask: batch_train_data[3],
                                                        self.y: batch_train_data[4]})
                    train_batch_loss, train_batch_err = sess.run([self.cost, self.err],
                                                                 feed_dict={self.eye_left: batch_train_data[0],
                                                                            self.eye_right: batch_train_data[1],
                                                                            self.face: batch_train_data[2],
                                                                            self.face_mask: batch_train_data[3],
                                                                            self.y: batch_train_data[4]})
                    train_loss += train_batch_loss / n_batches
                    train_err += train_batch_err / n_batches

                val_loss = 0.
                val_err = 0.

                for batch_val_data in next_batch(val_data, batch_size):
                    val_batch_loss, val_batch_err = sess.run([self.cost, self.err],
                                                             feed_dict={self.eye_left: batch_val_data[0],
                                                                        self.eye_right: batch_val_data[1],
                                                                        self.face: batch_val_data[2],
                                                                        self.face_mask: batch_val_data[3],
                                                                        self.y: batch_val_data[4]})

                    val_loss += val_batch_loss / val_n_batches
                    val_err += val_batch_err / val_n_batches

                train_loss_history.append(train_loss)
                train_err_history.append(train_err)

                val_loss_history.append(val_loss)
                val_err_history.append(val_err)

                if val_loss - min_delta < best_loss:
                    best_loss = val_loss
                    save_path = saver.save(sess, out_model, global_step=n_epoch)
                    print("Model saved in file: %s" % save_path)
                    n_incr_error = 0

                if n_epoch % print_per_epoch == 0:
                    print('Epoch %s/%s, train loss: %.5f, train error: %.5f, val loss: %.5f, val error: %.5f' %
                          (n_epoch, max_epoch, train_loss, train_err, val_loss, val_err))

                if n_incr_error >= patience:
                    print('Early stopping occured. Optimization Finished!')
                    return train_loss_history, train_err_history, val_loss_history, val_err_history

            return train_loss_history, train_err_history, val_loss_history, val_err_history


def train():
    train_data, val_data = load_data(TRAIN_DATA_PATH)

    train_data = prepare_data(train_data)
    val_data = prepare_data(val_data)

    # start = timeit.default_timer()

    et = EyeTracker()

    train_loss_history, train_err_history, val_loss_history, val_err_history = et.train(train_data, val_data,
                                                                                        lr=LEARNING_RATE,
                                                                                        batch_size=BATCH_SIZE,
                                                                                        max_epoch=MAX_EPOCH,
                                                                                        min_delta=1e-4,
                                                                                        patience=PATIENCE,
                                                                                        print_per_epoch=PRINT_PER_EPOCH,
                                                                                        out_model=SAVE_MODEL)

    if SAVE_LOSS:
        with open(SAVE_LOSS, 'w') as outfile:
            np.savez(outfile, train_loss_history=train_loss_history, train_err_history=train_err_history,
                     val_loss_history=val_loss_history, val_err_history=val_err_history)


def main():
    if TRAIN:
        train()


main()