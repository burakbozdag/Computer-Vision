from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(len(X)): # For each training sample
      scores = np.dot(X[i], W)

      # For improving numerical stability
      # https://cs231n.github.io/linear-classify/#softmax-classifier
      scores -= np.max(scores)
      probability_scores = np.exp(scores) / np.sum(np.exp(scores))

      for k in range(W.shape[1]): # For each class
        dW[:, k] += X[i] * (probability_scores[k] - int(y[i] == k))
        # Indicator is placed as in-line conditional statement
      
      loss -= np.log(probability_scores[y[i]])
    
    # Averaging - Regularization
    loss /= len(X)
    loss += reg * np.sum(W * W)

    # Averaging - Regularization
    dW /= len(X)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # X * W
    scores      = np.dot(X, W)

    # exp(X * W)
    dominator   = np.exp(scores)

    # sum(exp(X * W))
    # Reshaping is for getting an array of size of X.
    denominator = np.sum(np.exp(scores), axis=1).reshape((len(X), 1))
    
    softmax_scores = dominator / denominator

    # Getting correct scores 
    loss = np.sum(softmax_scores[range(len(X)), y])

    # Averaging - Regularization
    loss /= len(X)
    loss += reg * np.sum(W * W)

    # Gradient

    # Subtracting 1 from the correct class (indicator -> denominator--)
    softmax_scores[range(len(X)), y] -= 1

    # X * W
    dW = np.dot(np.transpose(X), softmax_scores)

    # Averaging - Regularization
    dW /= len(X)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
