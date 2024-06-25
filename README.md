# mnist-classifier
Neural network classifier for MNIST in Python with NumPy.

I wrote this because I wanted to test my understanding of how to implement a feed forward neural network. In particular, I wanted to write an implementation of back propagation. 

As a guide, I used an example from Deep Learning with Python, 2nd edition ([ref](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter02_mathematical-building-blocks.ipynb)), which is implemented with Keras. Here, I rewrote from scratch using vanilla Python along with NumPy for matrix operations.


## Run notes

Requires numpy. Also, download the MNIST dataset from http://yann.lecun.com/exdb/mnist/, extract, and place into a `mnist_archive` directory.

## Comments

Writing a neural network from scratch is informative as an exercise to build understanding and gain intuition. For solving actual problems, use libraries.

### Model structure
The examples in the MNIST dataset are 28 by 28 pixel greyscale images. Here, I vectorized them into vectors of dimensionality 28 * 28 = 784. The dataset labels are integers `0` through `9`.

The model here is a two layer feed forward neural network. The first layer is dimensionality 512 with ReLU activation. The second layer is dimensionality 10 with softmax activation.

### Forward propagation

I found forward propagation straight forward to understand.

Each layer computes an output vector $\pmb{y}$ from an input vector $\pmb{x}$ using weights $\pmb{W}$, biases $\pmb{c}$, and an activation function $g$.
$$\pmb{y}=g(\pmb{W}\pmb{x} + \pmb{c})$$

For understanding, I personally found it helpful to emphasize that the activation functions are vector functions, ie $\mathbb{R}^m \to \mathbb{R}^n$, even if the common activations are typically written on a per-element basis. eg. ReLU as $f(x)=\max(0,x)$

### Loss function

With this model, I used negative log likelihood as the loss function. This is also referred to as cross entropy.

### Back propagation

For me, back propagation was the trickiest piece of the neural network implementation to understand.

On first contact with the idea, I was not able to wrap my head around it. And, I found the notation for explanations of it to be daunting. Lots of partial derivatives, eg. $\frac{\partial u^{(n)}}{\partial u{(i)}}$. And what are these inverted triangles $\nabla_{\hat{y}}J$?

For an intuitive understanding of back propagation, I found the 3Blue1Brown videos ([ref](https://www.youtube.com/watch?v=Ilg3gGewQ5U), [ref](https://www.youtube.com/watch?v=tIeHLnjs5U8)) to be helpful.

Before I was able to parse through the mathematical description of back propagation though, I needed to learn the relevant notation. Apostol Vol II, Ch 8 Differential Calculus of Scalar and Vector Fields ([ref](https://archive.org/details/calculus-tom-m.-apostol-calculus-volume-2-2nd-edition-proper-2-1975-wiley-sons-libgen.lc/%5BCalculus%5D%20Tom%20M.%20Apostol%20-%20Calculus%2C%20Volume%202%2C%202nd%20Edition%20%28PROPER%29%202%281975%2C%20Wiley%20%26%20Sons%29%20-%20libgen.lc/page/243/mode/2up)) was sufficient for me.

With this background in place, I was able to follow taking derivatives of the loss function, the softmax function, an affine transformation, and so on. I found Eli Bendersky's [post](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative) on The Softmax function and its derivative to be helpful when going through this.

With back propagation, the idea is to compute partial derivatives of the loss function with respect to the model parameters, which can be done using the chain rule.

At this point, I was (finally) able to write a back propagation implementation.

For me, working through the computation of a gradient was necessary for understanding. Reading an implementation in code was not sufficient.

Also, looking at the resulting implementation, it's concise. Given that the analytic derivation is somewhat involved, it's neat that expressing it in code does not take many lines.

### Optimization

I tried training with mini batches with both stochastic gradient descent and RMSProp. (I followed the RMSProp algorithm as described in the Goodfellow Deep Learning book, [ref](https://www.deeplearningbook.org/contents/optimization.html)) For this dataset, RMSProp performed clearly better.

### Results

Running this project locally on an M1 Mac, I got an accuracy of over 90% on the test dataset. This shows that the implementation is correct

However, compared to an implementation with Keras this project is inefficient, and runs slowly. Initially, I attempted to write the project entirely in vanilla Python, ie representing matrices as Python lists of lists. This proved to be infeasibly inefficient. (For Python profiling, I found cProfile ([ref](https://docs.python.org/3/library/profile.html)) and snakeviz ([ref](https://jiffyclub.github.io/snakeviz/)) helpful.) Switching the matrix operations to NumPy substantially reduced runtime compared to that.

Also, models with different structure, such as CNN, have potential for higher accuracy.
