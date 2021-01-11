# Import the Packages

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Load the datasets

sess = tf.Session()
iris = datasets.load_iris()

# Classify the data into features & Labels

x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y==0 else -1 for y in iris.target])

# Split the dataset

train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace = False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Set the Batch size, placeholder & model variables

batch_size = 100
x_data = tf.placeholder(shape = [None, 2], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)
A = tf.Variable(tf.random_normal(shape = [2,1]))
b = tf.Variable(tf.random_normal(shape = [1,1]))

# Model output assign

model_output = tf.subtract(tf.matmul(x_data, A), b)

# maximum margin loss

l2_norms = tf.reduce_sum(tf.square(A))
alpha = tf.constant([0, 1])
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
loss = tf.add(tf.multiply(alpha, l2_norms), classification_term)

# Prediction & Accuracy

prediction + tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))
residuals = prediction - y_target

# Create interactive scalars 7 Histograms in the tensorboard

with tf.name_scope('Loss'):
    tf.summary.histogram('Histogram_Errors', accuracy)
    tf.summary.histogram('Histogram_Residuals', residuals)
    Loss_summary_OP = tf.summary.scalar('loss', loss[0])
summary_op = tf.summary.merge_all()

# declare optimizer & initialize model variables

my_opt = tf.train.GradientDescentoptimizer(0.01)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

# Running the session variables by feeding the train & test data

loss_vec = []
train_accuracy = []
test_accuracy = []

for i in range(5000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])

    _, train_loss, summary = sess.run([train_step, loss, summary_op], feed_dict={x_data: rand_x, y_target: rand_y})
    test_loss, test_resids = sess.run([loss, residuals],
                                      feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    loss_vec.append(train_loss)
    train_acc_temp = sees.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    train_accuracy.append(train_acc_temp)
    test_acc_temp = sees.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_accuracy.append(test_acc_temp)

    if (i + 1) % 50 == 0:
        print('Step #' + str(i + 1) + 'A = ' + str(sess.run(A)) + 'b = ' + str(sess.run(b)))
        print('Loss = ' + str(train_loss))
        print('Train Accuracy = ' + str(np.mean(train_accuracy)))
        print('Test Accuracy =' + str(np.mean(test_accuracy)))

log_writer = tf.summary.FileWriter('.......', sess.graph)
log_writer.add_summary(summary, i)

# Extract the coefficients & separate x values into setosa & non - setosa flowers

[[a1], [a2]] = sess.run(A)
[[b]] = sess.run(b)
slope = -a2/a1
y_intercept = b/a1
x1_vals = [d[1] for d in x_vals]
best_fit = []
for i in x1_vals:
    best_fit.append(slope*i+y_intercept)
    setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
    setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == 1]
    not_setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
    not_setosa_y = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]

# Graph Visualization

plt.plot(setosa_x, setosa_y, 'o', label = 'l, setosa')
plt.plot(not_setosa_x, not_setosa_y, 'o', label = 'Non - setosa')
plt.plot(x1_vals, best_fit, 'r-', label='Linear Seperator', linewidth = 3)
plt.ylim([0, 10])
plt.legend(loc = 'lower right')
plt.title('Sepal length vs Petal width')
plt.xlabel('Petal Width')
plt.ylabel('sepal length')
plt.show()
plt.plot(train_accuracy, 'k-', label='test accuracy')
plt.plot(train_accuracy, 'r-', label='test accuracy')
plt.title('train & Test set Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc = 'Sepal length')
plt.show()
plt.plot(loss_vec, 'k-')
plt.title('Sepal length vs Petal width')
plt.xlabel('Petal Width')
plt.ylabel('sepal length')
plt.show()