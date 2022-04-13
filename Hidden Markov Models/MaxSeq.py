# Problem 1 - Part (b)
# https://www.youtube.com/watch?v=xejm-z3sbWA
# I was able to get SmoothHMM to work, but I didn't understand how to take the max. I had to write this from scratch. Very inefficient.

import sys
import numpy as np

# find_most_likely_seq(init_prob, evid_seq, T, E)
#   1. Find the most likely sequence of states.
#
#   Parameters:
#     1. init_prob: Initial probabilities
#     2. evid_seq: Evidence seqeuence
#     3. T: T matrix
#     4. E: E matrix
def find_most_likely_seq(init_prob, evid_seq, T, E):
  # initialize prior to loop
  probs = []
  # predict
  x = np.multiply(init_prob, E[:,evid_seq[0]]) # initial probabs based on evidence and initial probs
  # extract maximum probabilities
  max_p = []
  for _ in x:
      max_p.append(round(np.amax(_), 2))
  # normalize
  max_p = max_p / np.linalg.norm(max_p, ord=1) # normalize because Viterbi example in lecture 6 normalized the t=1 probability
  probs.append(max_p)

  for evid in evid_seq[1:]: # first evid_seq used in initialization
    # predict
    x = np.multiply(T, E[:,evid])
    # update
    y = np.multiply(max_p, x)
    # extract maximum probabilities
    max_p = []
    for _ in y:
        max_p.append(round(np.amax(_), 3))
    # max_p_n = max_p / np.linalg.norm(max_p, ord=1) # not normalizing to align with lecture 6 example... why? 

    probs.append(max_p)

  probs = np.vstack(probs)

  max_seq = []
  for _ in reversed(probs):
    max_seq.append(abs(np.argmax(_)-1)) # because I reversed the T/F 1/0 earlier.

  return max_seq


# main()
#   1. Read users' input.
#   2. Find the most likely sequence of states.
def main():
  # Uncomment for testing: 0 = F, 1 = T
  # evid_seq = [0,1,0,1,0,1,0,0,1,0]
  # n = len(evid_seq)
  # T = np.matrix([[0.7, 0.3], [0.4, 0.6]])
  # E = np.matrix([[0.9, 0.1], [0.3, 0.7]])

  # Uncomment for submission
  n = int(sys.argv[1])
  evid_seq = []
  for i in range(n):
      evid_seq.append(int(sys.argv[i+2]))
  T = np.matrix([[0.7, 0.3], [0.4, 0.6]])
  E = np.matrix([[0.9, 0.1], [0.3, 0.7]])

  init_prob = np.array([0.5, 0.5])
  evid_seq = [abs(x - 1) for x in evid_seq]
  max_seq = find_most_likely_seq(init_prob, evid_seq, T, E)
  n = len(max_seq)
  for i in range(n):
      print(max_seq[i], end=" ")
  print()




if __name__ == "__main__":
  main()