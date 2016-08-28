import numpy as np
from random import shuffle
import pdb
from IPython import embed
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
  reg_loss = 0.0
  dW = np.zeros_like(W)
  N,D = X.shape
  D,C = W.shape
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #
  #y_pred = np.zeros((N,C))
  for i in range(N):
    y_pred = np.dot(X[i,:], W)
    max_y = max(y_pred)
    y_pred = y_pred - max_y
    y_pred = np.exp(y_pred) / sum(np.exp(y_pred))
    y_tmp = np.zeros((C))
    y_tmp[y[i]] = 1.0
    loss_softmax = - (y_tmp * np.log(y_pred))
    loss += np.sum(loss_softmax)
    dl = -(y_tmp - y_pred)
    dW += np.dot(X[i,:].reshape(D,1),dl.reshape(1,C))

    # divide the loss by batch size
  loss /=N
  dW /=N
  #pdb.set_trace()
  W_temp = W.reshape(1,-1)
  reg_loss = reg*np.sum(np.square(W_temp))
  # add regularization
  loss = loss + reg_loss
  return loss,dW


  # the derivate of cross entropy dp_i/dy_j is given by derivative of softmax
  # where p_i= e^y_i / sum_j (e^y_j)
  # is  dp_i/dy_j = p_i[i==j] - p_i*p_j
  # http://knet.readthedocs.io/en/latest/softmax.html

  # softmax loss is given by
  # W = sum_C (t_k * log(p_k))
  # gradient with respect to unnormalized y_j is
  #  dW/dp_j = sum_c (t_c * log(p_c)) where c = 1 to C
  #  dW/dp_j = sum_c (t_c * log(p_c/sum_l(p_l))) # sum_l(p_l) is not 1
  #         = sum_c (t_c* log(p_c) ) - sum_k (t_k * log sum_l(p_l))
  #          = sum_c (t_c * log(p_c)) - log sum_l(p_l) * sum_k(t_k)
  #          = sum_c (t_c * log(p_c)) - log sum_l(p_l) * 1
  #          = sum_c (t_c * log(p_c)) - log sum_l(p_l) * 1
  # dW/dp_j  = t_j/p_j - 1/sum_l(p_l)*d(sum_l(p_l)/dp_j
  # dW/dp_j  = t_j/p_j - 1/sum_l(p_l)*1
  # dW/dp_j  = t_j/p_j - 1/sum_l(p_l)*1 since sum_l(p_l)=1
  # dW/dp_j  = t_j/p_j - 1


  # dW/dy_j = sum_k( dW/dp_k * dp_k/dy_j)
  # dW/dy_j = sum_k (( t_k/p_k -1) * (p_k[k==j] - p_k*p_j))
  #   = (t_j/p_j -1) * (p_i[i==j] - p_i*p_j)
  #   = (t_j/p_j -1) * (p_j - p_j*p_j) for j=i
  #    = (t_j - p_j) *(1-p_j)
  #    = (t_j/p_j-1) *(-p_j*p_i) for i not j
  #     = (t_j - p_j)*(p_i)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  # dW will be same shape as W
  # expand dy = np.dot(dl)



def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  pdb.set_trace()
  if 1:
    y_pred = np.dot(X, W)
    max_y = max(y_pred)
    y_pred = y_pred - max_y
    y_pred = np.exp(y_pred) / sum(np.exp(y_pred))
    y_tmp = np.zeros((C))
    y_tmp[y[i]] = 1.0
    loss_softmax = - (y_tmp * np.log(y_pred))
    loss += np.sum(loss_softmax)
    dl = -(y_tmp - y_pred)
    dW += np.dot(X[i,:].reshape(D,1),dl.reshape(1,C))
    # divide the loss by batch size
  loss /=N
  dW /=N
  return loss,dW


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

