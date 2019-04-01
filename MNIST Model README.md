# Debugging Model with Tensorboard
## Visualizing your Model

This Guide will help you understand how you'll debug your model using Tensorboard.

## Dataset Used

The Dataset in this model is the famous `MNIST Dataset` which is a database of images of Hand written numbers.

The MNIST Dataset has 70,000 hand written images from zero(0) to nine(9) that have been size-normalized and centered in a square grid of pixels.

Each image is a 28X28X1 array of floating-point numbers representing grayscale intensities ranging from 0(black) to 1(white)

![mnist_images.png](https://www.dropbox.com/s/lilz3njhb80f7w1/mnist_images.png?dl=0&raw=1)

 Information: * name : MNIST * length : 70000

 Input Summary: * shape : (28, 28, 1) * range : (0.0, 1.0)

 Target Summary: * shape : (10,) * range : (0.0, 1.0)


### Loading the Dataset

Tensorflow has a pre-defined library of datasets and hence you can import the dataset from there. There are multiple options available like using urllib library and downloading the dataset on your own.

```sh
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

This will download the dataset to your current working directory.

# MNIST Model

This will be a `Convolutional Neural Network Model` with some convolutional, pooling and fully connected layers in it. There will also be a dropout layer.

`Current working model accruacy is upto 99%.`

The model will be simple,
![model (edited-Pixlr).jpg](https://www.dropbox.com/s/5qyeswnq5a4cirz/model%20%28edited-Pixlr%29.jpg?dl=0&raw=1)
  
For activation we use `ReLu` and we used `Adam Optimizer` to reduce our training loss.

# Visualizing our Model

Now after the model start to work we'll now start to visualize our model with the help of Tensorboard.

If you don't have any idea how to visualize using tensorboard then refer the `Documentation` provided with this `Github Repo`.

Now, to visualize things via `TensorBoard`, you first need to start its service. For that,

Run the `program` in your desired IDE and while the program is running open CMD and enable your virtualenv and redirect to the destination where your model is and type:

```sh
(tfdeeplearning) C:\Users\model\tensorboard --logdir="./output"
```

Note that `(tfdeeplearning)` is a virtualenv created on my pc, yours will be different!

Now the cmd will result with a link like this,

```sh
TensorBoard 1.13.0 at http://DESKTOP-I48HILA:6006 (Press CTRL+C to quit)
```
copy that desired `http://DESKTOP-I48HILA:6006` link and paste it into the web browser, now you'll see a window like this,


![main_graph.PNG](https://www.dropbox.com/s/gn79jxfovzoppvq/main_graph.PNG?dl=0&raw=1)

This is the main graph of our model.
You can now trace where the variable are going, from which function they are passing and finally going to which node.

You can also expand every node and see what is going on inside of the node.

The name to every node is given via `Name Scopes`

```sh
with tf.name_scopes(name):
```

Every Node has given a name so as to maintain an ease in visulaizing the graph.

Currently at this point if you only implemented the above code in your model then only graph tab will be giving the output none other, the reason is they all have to be implemented, we'll do it one by one:

## 1. Scalar Tab

This tab is for the scalar summary, now what is a scalar summary?
It is used to write a single scalar-valued tensor (like classificaion loss or accuracy value).

This is what we are going to use this tab for, now this will be implemented with the help of some more lines of code.

```sh
tf.summary.scalar(name = 'Cross_Entropy', tensor = cross_entropy)
tf.summary.scalar(name = 'Accuracy', tensor = acc)
```

These line will add the scalar summary with `Cross Entropy` and `Accuracy` function.

At this point we have our scalar summary,
Now there is a thing you every time you define a summary it results in an protocol buffer and you pass it through `tf.summary.FileWriter` but this would be really tedious if we did this for many summaries so for that what we have here is the `merge_all()` function which merges all the summaries into one, like

```sh
merged_summmary = tf.summary.merge_all()
```

This will merge all the summary in one and now you can pass it through FileWriter only one time.

```sh
if i%5 == 0:
    s = sess.run(merged_summmary, 
    feed_dict = {x: batch_x, y_true:batch_y,hold_prob:0.5})
    writer.add_summary(s, i)
```
This block of code runs the `merged_summary` and referesh the result for every 5 steps so as we get the scalar graphs precise and clear.

You can play with the number here according to your own choice.

** `Note Tensorboard will run the latest event file of there are multiple files in you r output directory then tensorboard will result to run the latest event file!`

Now after opening the web browser we'll see

SCAlAR Tab

![accuracy.png](https://www.dropbox.com/s/syzjjq4fm5qbnao/accuracy.png?dl=0&raw=1)

You can see we can now visualize our model that how it is training and the accuracy and loss via graph you can play in with smoothing and various tools to see what else can be done.

```Note: If you run the program then the event file will be overwritten and hence result in a unexpected summary graph. If you want a clean summary graph then change the directory to something else and this will create a new directory in your parent directory and will run separately.```

## 2. Image Tab

Now we come to the image tab here you'll see the training images on which you're training your data.

Just add these few lines,

```sh
tf.summary.image('input_x', x_image, 3)
```
and re-run your model and run the link again now you'll see,

![images.PNG](https://www.dropbox.com/s/dfxmmmoohijgvja/images.PNG?dl=0&raw=1)

You can now visualize the training images on which you're training your model on.

## 3.Histograms and Distribution Tab

##### Distributions
Visualize how data changes over time, such as the weights of a neural network.

##### Histograms
A fancier view of the distribution that shows distributions in a 3-dimensional perspective.

For this view we have to added few lines of code,

```sh
tf.summary.histogram("Weights", W)
tf.summary.histogram("Biases", b)
tf.summary.histogram("Activations", act)
```

The Weigths, biases and Activation function are now implmented with the histogram summary and will result in the following way,

Distribution Tab:
![distributions.PNG](https://www.dropbox.com/s/zig1bsldkryim2k/distributions.PNG?dl=0&raw=1)

Histogram Tab:
![histogram.PNG](https://www.dropbox.com/s/z5nsjbxissubud6/histogram.PNG?dl=0&raw=1)

## 4. Projector

It is a embedding visualizer and the coolest thing tensorboard has, here you can visulaize your embeddings seet according to your size and it wil return a 3D projection like this,

![proj.PNG](https://www.dropbox.com/s/fzdipnppvbwl547/proj.PNG?dl=0&raw=1)


You can see this is a 3D Projection of the MNIST Dataset and the various dots are the images floating aroung the 3D Plane, currently it is the `PCA(Principal Component Analysis)`, but you can also set it to `t-SNE(t-distributed stochastic neighbour embedding)` so as to get a more clearer view of the digits floating in the 3D space.

How to implement this,

The Projector needs to be implemented via few lines of code,

```sh
import os
from tensorflow.contrib.tensorboard.plugins import projector
# Create randomly initialized embedding weights which will be trained.
N = 10000 # Number of items (vocab size).
D = 784 # Dimensionality of the embedding.
LOG_DIR = os.getcwd()
embedding_var = tf.Variable(tf.random_normal([N,D]), name='word_embedding')
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = os.path.join(LOG_DIR + "/embedding", 'metadata.tsv')
```

Now this will run by,

```sh
projector.visualize_embeddings(writer, config)
```
Now, after this if you run the model it will not run the projector tab and will result in an error saying you did'nt saved your model!

we have to save our model for very few steps as the training continues, this will be done through,

```sh
saver = tf.train.Saver()
if i%500==0:
    saver.save(sess, os.path.join(LOG_DIR + ".\out", "model.ckpt"), i)
    
    sess.run(train, feed_dict = {x: batch_x, y_true: batch_y, hold_prob: 0.5})
    
```

Now for every 500 steps our model will be saved.

Now the projector tab will show the dersired result.

```Sometimes when you run the tensorboard there is no projector tab in the dashboard, what you  need to do is open the inactive cell and select projector from there. The reason to this is that projector need some time, some calcuations to takes place before it is intialized, running to quickly will result in an inactive projector tab```

Now here is a catch you see two files in the program i.e a `metadata.tsv` file and a `sprite_images.png` file which is linked to the projector. 

It is nothing but the `metadata.tsv` file has the stored index and label of each sample and the `sprite_images.png` has all the images stacked in a single big image file.

These files are stored in the `embedding files` folder you don't have to download it separately.

##### Finally the dataset has been implemented with the help of tensorboard which helped us to visualize our result and helped us to get a more clearer view that what is going on in this neural network.





