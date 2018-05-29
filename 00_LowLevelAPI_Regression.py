import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.INFO)

tf_version = tf.__version__
assert "1.4" <= tf_version, "TensorFlow r1.4 or later is needed"

learning_rate = 0.001
training_epochs = 15
batch_size = 100

# tf.enable_eager_execution()

PATH = "./"
PATH_DATASET = "./dataset"
FILE_TRAIN = PATH_DATASET + os.sep + "20180430_RC_Populated_DATA_Isolated_Train_Random.csv"
FILE_TEST = PATH_DATASET + os.sep + "IsolatedTEST_20180430_Random.csv"
FILE_PRACTICE = PATH_DATASET + os.sep + "IsolatedTEST_20180430_Random.csv"

# Create features
feature_names = ['vehicle', 'Mp01', 'Mp02', 'Mp03', 'Mp04', 'Mp05', 'Mp06', 'Mp07', 'Mp08', 'Mp09', 'Mp10', 'Mp11', 'Mp12']
vehicle = tf.feature_column.categorical_column_with_vocabulary_list('vehicle', ['Small', 'Mid', 'RV'])
feature_columns = [tf.feature_column.indicator_column(vehicle)]
for k in range(1,13):
    feature_columns.append(tf.feature_column.numeric_column(feature_names[k]))

def my_input_fn(file_path, repeat_count=1, shuffle_count=1):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[""], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]], field_delim=',')
        label = parsed_line[-1]
        del parsed_line[-1]
        features = parsed_line
        d = dict(zip(feature_names, features)), label
        return d

    with tf.name_scope('DATA_Feeding'):
        dataset = (tf.data.TextLineDataset(file_path)
            .skip(1)
            .map(decode_csv)
            .cache()
            .shuffle(shuffle_count)
            .repeat(repeat_count)
            .batch(32)
            .prefetch(1)  # Make sure you always have 1 batch ready to serve
        )
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

print(feature_columns)

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
         with tf.variable_scope(self.name):
             regularizer = tf.keras.regularizers.l2(l=0.01)
             initializer = tf.keras.initializers.glorot_normal(seed=None)

             # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
             self.training = tf.placeholder(tf.bool)

             #features = tf.parse_example(..., features=tf.feature_column.make_parse_example_spec('serialized', feature_columns))
             #dense_tensor = input_layer(features, feature_columns)

             # input place holders
             self.X = tf.placeholder(tf.float32, [None, 15])
             self.Y = tf.placeholder(tf.float32, [None, 1])

             #input_layer = tf.feature_column.input_layer(features, feature_columns)
             input_layer = tf.feature_column.input_layer(self.X, self.Y)
             h1 = tf.layers.Dense(units=30,
                                  activation=tf.nn.relu,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  activity_regularizer=None)(input_layer)
             h2 = tf.layers.Dense(units=30,
                                  activation=tf.nn.relu,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  activity_regularizer=None)(h1)
             h3 = tf.layers.Dense(units=30,
                                  activation=tf.nn.relu,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  activity_regularizer=None)(h2)
             dropout4 = tf.layers.dropout(inputs=h3, rate=0.5, training=self.training)

             # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
             self.logits = tf.layers.dense(inputs=dropout4, units=10)

         '''
         predictions = {'Squeeze': tf.squeeze(self.logits, 1),  # Squeeze is result value
                        'MP01': features['Mp01'],
                        'MP02': features['Mp02'],
                        'MP03': features['Mp03'],
                        'MP04': features['Mp04'],
                        'MP05': features['Mp05'],
                        'MP06': features['Mp06'],
                        'MP07': features['Mp07'],
                        'MP08': features['Mp08'],
                        'MP09': features['Mp09'],
                        'MP10': features['Mp10'],
                        'MP11': features['Mp11'],
                        'MP12': features['Mp12'],
                        'vehicle_type': features['vehicle']
                        }
         '''

         # define cost/loss & optimizer
         #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
         self.cost = tf.losses.mean_squared_error(self.Y, tf.squeeze(self.logits, 1)) # predictions['Squeeze'])

         self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
         #self.optimizer = tf.train.AdamOptimizer(learning_rate, name="My_Optimizer")
         #self.train_op = self.optimizer.minimize(loss=average_loss, global_step=tf.train.get_global_step())
         correct_prediction = tf.metrics.recall(self.Y - tf.squeeze(self.logits, 1)) #predictions['Squeeze'])

         #self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
         self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # It means RMS value

    def predict(self, x_test, training=False):
        return self.sess.run(tf.squeeze(self.logits, 1), feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: training})



# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = 10 #int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = my_input_fn(FILE_TRAIN, 500, 256) # mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))