# CSE 576 Homework 4 #

Welcome, it's time for neural networks! Needless to say, run

```
git pull
```

Then `python -m src.main test hw4` and/or `python tryhw4.py`.

## 1. Implementing neural networks ##

### 1.1 Activation functions ###

An important part of machine learning, be it linear classifiers or neural networks, is the activation function you use. 

#### TO DO ####
Fill in `def activate_matrix(m, a)` to modify `m` to be `f(m)` applied elementwise where the function `f` is given by what the activation `a` is.

Remember that for our softmax activation we will take e^x for every element x, but then we have to normalize each element by the sum as well. Each row in the matrix is a separate data point so we want to normalize over each data point separately.

### 1.2 Taking gradients ###

As we are calculating the partial derivatives for the weights of our model we have to remember to factor in the gradient from our activation function at every layer. We will have the derivative of the loss with respect to the output of a layer (after activation) and we need the derivative of the loss with respect to the input (before activation). To do that, we take each element in our delta (the partial derivative) and multiply it by the gradient of our activation function.

Normally, to calculate `f'(x)` we would need to remember what the input to the function, `x`, was. However, all of our activation functions have the nice property that we can calculate the gradient `f'(x)` only with the output `f(x)`. This means we have to remember less stuff when running our model.

The gradient of a linear activation is just 1 everywhere. The gradient of our softmax will also be 1 everywhere because we will only use the softmax as our output with a cross-entropy loss function, for an explaination of why see [here](https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa) or [here](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/#softmax-and-cross-entropy-loss).

The gradient of our logistic function is discussed [here](https://en.wikipedia.org/wiki/Logistic_function#Derivative):

    f'(x) = f(x) * (1 - f(x))

Figure out on your own what `f'(x)` will be, given `f(x)` is for RELU and Leaky RELU.

As discussed in the backpropagation slides, we just multiply our partial derivative by `f'(x)` to backpropagate through an activation function. 

#### TO DO ####
Fill in `def gradient_matrix(m, a, d)` to multiply the elements of `d` by the correct gradient information where `m` is the output of a layer that uses `a` as it's activation function.

### 1.3 Forward propagation ###

Now we can fill in how to forward-propagate information through a layer. First check out our layer struct:

    typedef struct {
        matrix in;              // Saved input to a layer
        matrix w;               // Current weights for a layer
        matrix dw;              // Current weight updates
        matrix v;               // Past weight updates (for use with momentum)
        matrix out;             // Saved output from the layer
        ACTIVATION activation;  // Activation the layer uses
    } layer;

During forward propagation we will do a few things. We'll multiply the input matrix by our weight matrix `w`. We'll apply the activation function `activation`. We'll also save the input and output matrices in the layer so that we can use them during backpropagation. But this is already done for you.

#### TO DO ####
Fill in the TODO in `def forward_layer(l, in_matrix)`. You will need the matrix multiplication function `matrix_mult_matrix(a, b)` provided by the matrix helper library.

Using matrix operations we can batch-process data. Each row in the input `in` is a separate data point and each row in the returned matrix is the result of passing that data point through the layer.

### 1.4 Backward propagation ###

We have to backpropagate error through our layer. We will be given `dL/dy`, the derivative of the loss with respect to the output `y` of the layer. The output `y` is given by `y = f(xw)` where `x` is the input, `w` is our weights for that layer, and `f` is the activation function. What we want to calculate is `dL/dw` so we can update our weights and `dL/dx` which is the backpropagated loss to the previous layer. 

#### TO DO ####
Fill in `def backward_layer(l, delta)`

#### 1.4.1 Gradient of activation function ####

First we need to calculate `dL/d(xw)` using the gradient of our activation function. Recall:

    dL/d(xw) = dL/dy * dy/d(xw)
             = dL/dy * df(xw)/d(xw)
             = dL/dy * f'(xw)

Use your `gradient_matrix(m, a, d)` function to change delta from `dL/dy` to `dL/d(xw)`

#### 1.4.2 Derivative of loss w.r.t. weights ####

Next we want to calculate the derivative with respect to our weights, `dL/dw`. Recall:

    dL/dw = dL/d(xw) * d(xw)/dw
          = dL/d(xw) * x

but remember from the slides, to make the matrix dimensions work out right we acutally do the matrix operiation of `xt * dL/d(xw)` where `xt` is the transpose of the input matrix `x`.

In our layer we saved the input as `l.in`. Calculate `xt` using that and the matrix transpose function in our helper code, `transpose_matrix(m)`. Then calculate `dL/dw` and save it into `l.dw`. We'll use this later when updating our weights.

#### 1.4.3 Derivative of loss w.r.t. input ####

Next we want to calculate the derivative with respect to the input as well, `dL/dx`. Recall:

    dL/dx = dL/d(xw) * d(xw)/dx
          = dL/d(xw) * w

again, we have to make the matrix dimensions line up so it actually ends up being `dL/d(xw) * wt` where `wt` is the transpose of our weights, `w`. Calculate `wt` and then calculate dL/dx. This is the matrix we will return.

### 1.5 Weight updates ###

After we've performed a round of forward and backward propagation we want to update our weights. Remember from class with momentum and decay our weight update rule is:

    Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    w_{t+1} = w_t + ηΔw_t

We'll be doing gradient ascent (the partial derivative component is positive) instead of descent because we'll flip the sign of our partial derivative when we calculate it at the output. We saved `dL/dw_t` as `l.dw` and by convention we'll use `l.v` to store the previous weight change `Δw_{t-1}`.

Calculate the current weight change as a weighted sum of `l.dw`, `l.w`, and `l.v`. The function `axpy_matrix(a, x, y)` will be useful here, it calculates the result of the operation `ax+y` where `a` is a scalar and `x` and `y` are matrices. Save the current weight change in `l.v` for next round.

Finally, apply the weight change to your weights by adding a scaled amount of the change based on the learning rate.

#### TO DO ####
Fill in `def update_layer(l, rate, momentum, decay)`.

## 1.6 Read through the rest of the code ##

Check out the remainder of the functions which string layers together to make a model, run the model forward and backward, update it, train it, etc. Notable highlights:

#### `make_layer(input_dim, output_dim, activation)` ####

Makes a new layer that takes `input` number of inputs and produces `output` number of outputs. Note that we randomly initialize the weight vector, in this case with a uniform random distribution between `[-sqrt(2/input), sqrt(2/input)]`. The weights could be initialize with other random distributions but this one works pretty well. This is one of those secret tricks you just have to know from experience!

#### `accuracy_model(m, d)` ####

Will be useful to test out our model once we are done training it.

#### `cross_entropy_loss(y, p)` ####

Calculates the cross-entropy loss for the ground-truth labels `y` and our predictions `p`. Cross-entropy loss is just negative log-likelihood (we discussed log-likelihood in class for logistic regression) for multi-class classification. Since we added the negative sign it becomes a loss function that we want to minimize. Cross-entropy loss is given by:

    L = Σ(-y_i log(p_i))

for a single data point, summing across the different classes. Cross-entropy loss is nice to use with our softmax activation because the partial derivatives are nice. If the output of our final layer uses a softmax: `y = σ(wx)` then:

    dL/d(wx) = (y - truth)

During training, we'll just calculate `dL/dy` as `(y - truth)` and set the gradient of the softmax to be 1 everywhere to have the same effect, as discussed earlier. This avoids numerical instability with calculating the "true" `dL/dy` and `dσ(x)/dx` for cross-entropy loss and the softmax function independently and multiplying them together.

#### `train_model(...)` ####

Our training code to implement SGD. First we get a random subset of the data using `random_batch(d, n)`. Then we run our model forward and calculate the error we make `dL/dy`. Finally we backpropagate the error through the model and update the weights at every layer.

## 2. Experiments with MNIST ##

Download the [MNIST PNG dataset from Kaggle](https://www.kaggle.com/datasets/ben519/mnist-as-png) and extract it into the repository root (the folder containing `tryhw4.py`). The extracted `mnist-png` directory should contain `train` and `test`.

From the repository root, use Bash to generate the image lists:

```bash
find mnist-png/train -type f -name '*.png' > mnist.train
find mnist-png/test -type f -name '*.png' > mnist.test
```

Run `python tryhw4.py` from the same directory so it can find both the lists and the image paths they contain.

Answer the questions in this section in a document which you later need to convert to a PDF and submit as `hw4.pdf`. Answers can be short, 1-2 sentences.

### 2.1 Train a linear softmax model ###

Check out `tryhw4.py` to see how we're actually going to run the machine learning code you wrote. There are a couple of example models in here for softmax regression and neural networks. Run the file with: `python tryhw4.py` to train a softmax regression model on MNIST. You will see a bunch of numbers scrolling by, this is the loss calculated by the model for the current batch. Hopefully over time this loss goes down as the model improves. Note: the data loading will take some time.

After training, our python script will calculate the accuracy the model gets on both the training dataset and a separate testing dataset.

#### 2.1.1 Question ####

What are the training and test accuracy values you get? Why might we be interested in both training accuracy and testing accuracy? What do these two numbers tell us about our current model?

#### 2.1.2 Question ####

Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?

#### 2.1.3 Question ####

Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?

### 2.2 Train a neural network ###

Now change the training code to use the neural network model instead of the softmax model.

#### 2.2.1 Question ####

Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed. How well do they perform? What's best?

#### 2.2.2 Question ####

Using the same activation, find the best (power of 10) learning rate for your model. What is the training accuracy and testing accuracy?

#### 2.2.3 Question ####

Right now the regularization parameter `decay` is set to 0. Try adding some decay to your model. What happens, does it help? Why or why not may this be?

#### 2.2.4 Question ####

Modify your model so it has 3 layers instead of two. The layers should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 3000 iterations instead of 1000. Look at the training and testing accuracy for different values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?

## 3. Experiments with CIFAR ##

Download the [CIFAR-10 PNG dataset from Kaggle](https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders) and extract it into the repository root. Keep both levels of the `cifar10` directory so the image folders are at `cifar10/cifar10/train` and `cifar10/cifar10/test`.

From the repository root, use Bash to generate the image lists:

```bash
find cifar10/cifar10/train -type f -name '*.png' > cifar.train
find cifar10/cifar10/test -type f -name '*.png' > cifar.test
```

The loader uses each image's parent directory as its class label.

### 3.1 Train on CIFAR ###

Change the image-list arguments in `tryhw4.py` to `cifar.train` and `cifar.test`. Run the script from the repository root to train your 3-layer neural network on CIFAR. You may have to fiddle with the learning rate to get it to train well.

#### 3.1.1 Question ####

What is the best training accuracy and testing accuracy? Summarize all the hyperparameter combinations you tried.

## Turn it in ##

Turn in your `classifier.py` and `hw4.pdf` on canvas under Homework 4.
