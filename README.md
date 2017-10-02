# Common adversarial noise for fooling a neural network

These days deep Neural Networks (NN) show exceptional performance on speech and visual recognition tasks. These systems are still considered a black box without deep understanding why they perform in such a manner. This lack of understanding makes NNs vulnerable to specially crafted adversarial examples - inputs with small perturbations that make the model misclassify. In this paper, we generated adversarial examples that will fool a NN used for classifying handwritten digits. We start by generating additive adversarial noise for each image, then we craft a single adversarial noise that misclassifies different members of the same class.

# Introduction
Artificial intelligence (AI), especially its subfield of machine learning (ML) became very popular over the last few years. A lot of experts today are saying that the next big breakthrough in AI will mark the new industrial revolution. There is more and more application of AI in many different industries as time goes by, which indicates that we should be very concerned about safety.

In the 2014, Szegedy et al. [1][1] discovered that you could make Adversarial Examples (AE) by adding carefully crafted small perturbations to the input and fool several ML models, including NN that have high accuracy on previously unseen data. Goodfellow et al. [2] explained this as a result of the models being too linear. With that kind of attack you could make an autonomous vehicle misclassify a stop sign, fool an intrusion-detection system, fool an identification system and so on. Most popular approaches for crafting AE are: the fast gradient sign method [2, 3] and Papernot et al. method [6]. On the other side, best strategies that are currently known for making models more robust to adversarial examples are: adversarial training [1, 2] and defensive distillation [7]. It turns out that none of these are fully successful and that an attacker can always find a new way to synthesize AE to overcome these defenses. One interesting type of attack that overcomes these defenses without knowing the models internals is the black-box attack [4]. It is also shown that it is possible to make AE that are robust to rescaling, translation and rotation [9]. Generating AE and defending against them is still an open research topic.

Goals of attackers are diverse. Some simply want the model to make a mistake while others aim to achieve class-targeted misclassification. There are relevant scenarios where an attacker can get a time limited access to the model internals. In those scenarios, he can create a common adversarial noise that can later be used to make the model misclassify any member of a class. We explore that idea later in the paper.

In section 2, we describe the NN that we fool with AEs. In section 3, we present algorithms for generating adversarial noise and show their results. In the last section, we conclude the paper with a few of our thoughts.

# Neural Network
In recent years Neural Networks (NN) have become widely popular due to their universal character (used for supervised, unsupervised and reinforcement learning) and increasing affordability of powerful distributed systems capable of processing large amount of data. A NN is composed of layers which contain units called neurons. Each neuron is parametrized by its weight vector w_ij. The NN also has a nonnegative cost function J(W) assigned to it. During the learning phase, a network is given training data, input-output pairs (x,y), and the goal is to adjust the set of weights W in a way that minimizes the cost function J(W) by using some optimization algorithm.

First, we need a classifier that we want to fool. We use a specific type of NN - Convolutional Neural Network (CNN) to classify images of handwritten digits. We used the famous MNIST dataset [11] which contains 70,000 28x28 pixel images of digits. Each pixel has one of 256 intensity values evenly distributed between 0 and 1. We divide the data into the training set (55,000 images), validation set (5,000 images) and the test set (10,000 images).

We trained our CNN on the training set and evaluated it on the test set. The validation set will be used later for crafting the adversarial noise shared amongst the whole class. The network has the architecture described in Table I.

The network was implemented in Python 3.5, using the tensorflow library. For the cost function we used cross-entropy. We minimized it using the Adam optimizer [5]. The training lasted 20,000 iterations and the weights were updated using a random batch of size 50 in each iteration. After the training, the NN had an accuracy of 99.19% on the test set.

# Adversarial Noise


[1]: https://github.com/Maki94/cnn_adv_examples/blob/master/literature/%5B1%5D%20Intriguing%20properties%20of%20neural%20networks.pdf
