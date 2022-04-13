# Problem 1 - Part (a)

from mimetypes import init
import sys
import numpy as np

# forward(pri_prob, evid_seq, T, E)
#   1. Calculate the sequence of forward probabilities.
#
#   Parameters:
#     1. init_prob: Initial probabilities
#     2. evid_seq: Evidence seqeuence
#     3. T: T matrix
#     4. E: E matrix
def forward(init_prob, evid_seq, T, E):
  prob = init_prob
  probs = []

  for evid in evid_seq:
    # Prediction
    prob = np.dot(prob, T)
    prob = np.asarray(prob).reshape(-1) # convert back to array vs 1D Matrix
    # Update
    prob = np.multiply(np.array(E)[:,evid], prob) # Update prediction with evidence
    prob = prob / np.linalg.norm(prob, ord=1) # normalize

    # attempting one step
    # prob = np.dot(prob, np.dot(T, E))
    # prob = np.dot(np.dot(prob, T), E.T)
    probs.append(prob)
  # Note: if you need to remove the 'array' from the print out and return only python list: https://stackoverflow.com/questions/62447005/output-of-list-append-prints-word-array-after-each-iteration
  probs = np.vstack(probs)
  return probs

# backward(last_prob, evid_seq, T, E)
#   1. Calculate the sequence of backward probabilities.
#
#   Parameters:
#     1. last_prob: Last probabilities
#     2. evid_seq: Evidence seqeuence
#     3. T: T matrix
#     4. E: E matrix
def backward(last_prob, evid_seq, T, E):
  prob = last_prob
  probs = []

  for evid in reversed(evid_seq):
    # Prediction
    prob = np.multiply(prob, np.array(E)[:,evid])
    prob = np.asarray(prob).reshape(-1) # convert back to array vs 1D Matrix

    # Update
    prob = np.dot(prob, np.array(T)) # how to avoid needing to convert from matrix to array??
    # prob = prob / np.linalg.norm(prob, ord=1) #  no normalize needed until smoothing

    probs.append(prob)
  # probs.reverse()
  probs = np.vstack(probs)
  return probs


# smooth(init_prob, last_prob, evid_seq, T, E)
#   1. Calculate the smoothed estimates, given the sequence of evidence.
#
#   Parameters:
#     1. init_prob: Initial probabilities
#     2. last_prob: Last probabilities
#     3. evid_seq: Evidence seqeuence
#     4. T: T matrix
#     5. E: E matrix
def smooth(init_prob, last_prob, evid_seq, T, E):
  f_dists = forward(init_prob, evid_seq, T, E)
  b_dists = backward(last_prob, evid_seq, T, E)

  s_dists = np.multiply(f_dists, b_dists)
  s_dists_norm = []

  # print(f_dists)
  # print(b_dists)
  # print(s_dists)


  for dist in s_dists:
    dist = dist / np.linalg.norm(dist, ord=1) #  normalize
    s_dists_norm.append(dist)

  # print(s_dists_norm)
  return s_dists_norm


# main()
#   1. Read users' input.
#   2. Calculate the smoothed estimates.
def main():
  # Uncomment for testing: 0 = F, 1 = T
  # evid_seq = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  # # evid_seq = [1, 1]
  # T = np.matrix([[0.7, 0.3], [0.3, 0.7]])
  # E = np.matrix([[0.9, 0.1], [0.2, 0.8]])
  # n = len(evid_seq)
  
  # Uncomment for submission
  n = int(sys.argv[1])
  evid_seq = []
  for i in range(n):
      evid_seq.append(int(sys.argv[i+2]))

  T = np.matrix([[0.7, 0.3], [0.4, 0.6]])
  E = np.matrix([[0.9, 0.1], [0.3, 0.7]])


  init_prob = np.array([0.5, 0.5])
  last_prob = np.array([1, 1])
  evid_seq = [abs(x - 1) for x in evid_seq]
  ans = smooth(init_prob, last_prob, evid_seq, T, E)
  n = len(ans)
  for i in range(n):
      print(ans[i][0], end=" | ")
  print()




if __name__ == "__main__":
  main()