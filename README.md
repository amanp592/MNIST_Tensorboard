# Tensorboard

TensorFlow is an open-source machine learning library used for research and production. TensorFlow offers APIs for beginners and experts to develop applications for desktop, mobile, web, and cloud. See the sections below to get started. 

### Installation

Tensorboard comes with Tensorflow by default.
So if you will install tensorflow, tensorboard will automatically be installed.

Guide to [install](https://www.tensorflow.org/install/) Tensorflow 

# Working

TensorBoard is created as a way to help us understand the flow of tensors in your model so that we can debug and optimize it. It is generally used for two main purposes:

1. Visualizing the Graph

2. Writing Summaries to Visualize Learning

## Visualizing  the Graph

Tensorflow computational graph can be extremely complicated to understand, but through Tensorboard you can visualize your graph to see what's going on.

For this you need some lines of code to work.

For example:
```sh
import tensorflow as tf
# create graph
a = tf.add(1, 2)
b = tf.mulitply(a, 4)
c = tf.add(4, 5)
d = tf.mulitply(c, 10)
e = tf.divide(b, d)
# launch the graph in a session
with tf.Session() as sess:
print(sess.run(c))
```

This is the basic script for a tensorflow program.

### Now Tensorboard comes into play.

To let Tensorboard work we add [Summarwriter](https://www.tensorflow.org/api_docs/python/tf/summary) to the end of our code, this in return create a directory in your given directory and will have tensorboard graphs information.

```sh
with tf.Session() as sess:
    writer = tf.summary.FileWriter("output", sess.graph)
    print(sess.run(c))
    writer.close()
```
Now our code looks like this:

```sh
import tensorflow as tf
tf.reset_default_graph() # This command resets the global default graph
# create graph
a = tf.add(1, 2)
b = tf.mulitply(a, 4)
c = tf.add(4, 5)
d = tf.mulitply(c, 10)
e = tf.divide(b, d)
# launch the graph in a session
with tf.Session() as sess:
    writer = tf.summary.FileWriter("output", sess.graph)
    print(sess.run(c))
    writer.close()
```

Now the code snippet makes a directory named `output` in the give directory.

*** Note running multiple times will result in making mulitple files in the given directory bu tensorboard will only run the latest file.

Now to work with tensorboard you have to open Command Prompt(Terminal in Linux/Mac) 

In CMD:
First go to the directory where the "output" folder is and then write

```sh
tensorboard --logdir="./ouput"
```
This will genearte a link in the command line with a message like

```sh
TensorBoard 1.13.0 at http://DESKTOP-I48HILA:6006 (Press CTRL+C to quit)
```
 
Now copy the link and run it in a browser and the window will open

![tensorboard.PNG](https://www.dropbox.com/s/qh6m1zo8iimp1r4/tensorboard.PNG?dl=0&raw=1)


This is Tensorboard Web GUI, here you can visualize your graph and can also debug your code.

## Writing Summaries

`Summary` is a special operation TensorBoard that takes in a regular tensor and outputs the summarized data to your disk.

There are three four types of Summaries:

1.  `tf.summary.scalar`
2.  `tf.summary.histogram`
3.  `tf.summary.image`

`tf.summary.scalar` is used to write a single scalar-valued tensor (like a classificaion loss or accuracy value)

`tf.summary.image` is used to plot images (like input images of a network, or generated output images of an autoencoder or a GAN)

`tf.summary.histogram` is used to plot histogram of all the values of a non-scalar tensor (can be used to visualize weight or bias matrices of a neural network)
