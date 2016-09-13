import numpy as np
from random import shuffle
import pdb
from IPython import embed
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
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    # if wrong class have scores < correct_class_score - delta (margin_loss) then no error
    # otherwise error is given bu (score - correct_class_score) + delta (we want a margin_loss of delta)
    if 0:
      for j in xrange(num_classes):
        if j==y[i]:
          continue
        margin = scores[j] - correct_class_score + 1
        if margin>0:
          loss+=margin
          dW[:,j] +=X[i,:]
          dW[:,y[i]] -= X[i,:]
    if 1:
      margin_loss = np.maximum(0, scores - correct_class_score + 1) # note delta = 1
      margin_loss[y[i]] = 0 # zero out the loss due to correct class.
      loss += np.sum(margin_loss)
      # this is saying only account 1 for places where margin (otherwise zero)
      margin_gradient = np.asarray(margin_loss>0, dtype=np.float32) #(1*np.asarray(margin_loss>0, np.float32))
      # for the correct class set the margin error gradient to zero
      margin_gradient[y[i]] = 0
      # margin_gradient error is then computed by summing the classes we got wrong.
      margin_gradient[y[i]] = -np.sum(margin_gradient)
      dW += np.dot(X[i].reshape(-1,1), margin_gradient.reshape(1,-1)) #dd_pos + dd_neg

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /=num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW /=num_train
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  D, C = W.shape
  N, D = X.shape
  delta = 1
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  y_tmp = np.dot(X,W)
  rows = range(N)
  class_scores = y_tmp[rows,y]
  margin_loss = np.max(0, y_tmp - class_scores + delta)
  margin_loss[rows,y] = 0
  loss = np.sum(margin_loss, axis=1)
  loss = np.sum(loss, axis=0)/N
  loss += 0.5 *reg *np.sum(W*W)
  dW = np.dot(X.T, margin_loss)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
