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
  D, C = W.shape
  N, D = X.shape
    
  # score
  scores = np.zeros((N, C))
  for xi in range(N):
    for ci in range(C):
      for di in range(D):
        scores[xi, ci] += X[xi, di] * W[di, ci]
  
  # probability
  probs = np.zeros((N, C))
  for xi in range(N):
    max_s = scores[xi, 0]
    for ci in range(1, C):
      max_s = max(max_s, scores[xi, ci])
    
    exp_sum = 0
    for ci in range(C):
      probs[xi, ci] = np.exp(scores[xi, ci] - max_s)
      exp_sum += probs[xi, ci]
    for ci in range(C):
      probs[xi, ci] /= exp_sum

  # loss
  for xi in range(N):
    loss += -np.log(probs[xi, y[xi]]) / N
    
  # dW
  for di in range(D):
    for ci in range(C):
      for xi in range(N):
        dW[di, ci] += probs[xi, ci] * X[xi, di]
        if ci == y[xi]:
          dW[di, ci] -= X[xi, di]
      dW[di, ci] = dW[di, ci] / N + reg * W[di, ci]

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  N = X.shape[0]
  C = W.shape[1]
  
  # loss
  scores = X.dot(W)
  exps = np.exp(scores - scores.max(axis=1).reshape(-1, 1))
  probs = exps / exps.sum(axis = 1).reshape(-1, 1)
  loss = -np.log(probs[range(N), y]).sum() / N
  
  # gradient
  probs[range(N), y] -= 1
  dW = X.T.dot(probs) / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

