from builtins import range
import numpy as np

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                #################################################################
                # In this part, in addition to the loss value calculation,
                # I perform gradient descent calculation for each train sample and corresponding classes.

                # For j = y[i] (correct class), the gradient formula is:
                # 1(scores[j] - correct_class_score + 1 > 0) * -X[i]
                dW[:, y[i]] = dW[:, y[i]] - X[i]
                # Simply, if margin > 0, we subtract X[i] from the y[i]'th column of the dW.

                # For j != y[i] (incorrect class), the gradient formula is:
                # 1(scores[j] - correct_class_score + 1 > 0) * X[i]
                dW[:,  j  ] = dW[:,  j  ] + X[i]
                # It means that if margin > 0, we add X[i] to the j'th column of the dW.
                #################################################################

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # https://cs231n.github.io/optimization-1/
    # I formulated gradient descent calculation with the help of this page.
    # I edited the loss function loop above as stated in the TODO section.
    # Related statements are given above as comment lines

    # Taking average of the gradient matrix values (because the average of loss values were taken)
    dW /= num_train

    # Add regularization to the gradient.
    # I almost forgot to add this line but a comment line in the notebook helped me :)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Using advantages of vectorization and broadcasting, these calculations can be faster.

    # scores: X * W
    scores = np.dot(X, W)

    # correct_class_scores: Scores that are selected by the corresponding correct class.
    correct_class_scores = scores[range(len(scores)), y]

    # https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
    # https://numpy.org/doc/stable/reference/generated/numpy.matrix.html
    # Margins are calculated with vectorization of score values. (np.matrix)
    # Transpose operation is applied for the broadcasting.
    margins = np.maximum(
        0,
        scores - np.transpose(np.matrix(correct_class_scores)) + 1 # note delta = 1
        )
    
    # Correct classes' margins must be zero before summation.
    margins[range(len(X)), y] = 0
    
    # Sum of values along rows, finally the mean value is calculated.
    loss = np.mean(np.sum(margins, axis=1))

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Margins can be used as intermediate values for the gradient descent.

    # If margin is greater than zero, it means that we can add X[i] at the related index.
    margins[margins > 0] = 1
    # Margin values greater than zero are set to 1. (For counter effect when taking sum)

    # Sum of the rows are taken which will act as counter for X[i].
    sum = np.sum(margins, axis=1)

    # Values of correct classes must be set to negative value.
    # Remember from the naive implementation above (j = y[i])
    margins[range(len(X)), y] = -np.transpose(sum)

    # When we are multiplying X with margins, margins matrix will now act as a counter for related rows.
    # This will result in gradient values in a matrix. 
    dW = np.transpose(X) * margins

    # Averaging
    dW /= len(X)

    # Regularizing
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
