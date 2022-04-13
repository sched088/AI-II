# Problem 2 - Part (b)

from statistics import variance
import sys
import numpy as np

# generate_samples(init_prob, evid_seq, T, E, n, size)
#   1. Randomly generate a number of samples.
#
#   Parameters:
#     1. init_prob: Initial probabilities
#     2. evid_seq: Evidence seqeuence
#     3. T: T matrix
#     4. E: E matrix
#     5. n: Number of samples to be generated
#     6. size: Number of states in a sample
def generate_samples(init_prob, evid_seq, T, E, n, size): # why do we need size? Can't we just go off of the length of the evidence?
  dict = {}
  for _ in range(n-1):
    w = 1 # initialize weight
    x = [] # node states
    # prob = np.dot(init_prob, T) # not needed for this given T
    # x.append(np.random.choice([0, 1])) # Set start value P(0.5) - prior || 0 = T 1 = F || need to convert at end // removed to utilize init_prob

    # set R0
    p = np.random.random()       
    if p < init_prob[0]: # 0.5 chance of rain
      x.append(1) # rain
    else:
      x.append(0) # no rain

    # loop R1:R10    
    for evid in evid_seq:
      # non-evidence
      p = np.random.random() # sample for new state (Ri+1) given current state (Ri)      
      if p < T[0,0]: # 0.7 chance of same ## note: shortcut here because of equal probs of given T
        x.append(x[-1]) # stay same
      else:
        x.append(abs(x[-1] - 1)) # switch states

      # evidence
      w = np.multiply(w, np.array(E)[[x[-1]],evid]) # update weight to P(State (t), Evidence (t))
      x.append(evid)

    x = x[1:] # remove R0 ##
    x = [abs(i - 1) for i in x] # convert x back to proper T/F 1/0 reference
    x = tuple(x)
    if x in dict.keys():
      dict[x] += w[0] 
    else:
      dict.update({x : w[0]})
  # print(x)
  # print(dict)
  return dict


# cal_weighted_prob_from_samples(samples)
#   1. Calculate the probability, based on the samples.
#
#   Parameters:
#     1. samples: Generated samples.

''' # I would think to check to see if the key matches 
the given evidence sequence to decide how to come up 
with the probability but we only get samples. 
Instead assuming most likely is true evidence sequence (1).

Update: We looking where R10 = T across all iterations'''

def cal_weighted_prob_from_samples(samples):
  w_probs = []
  r_prob = 0
  for key in samples:
    if key[-2] == 1: # if R10 is true (note we changed notation back to T = 1 F = 0) -2 because -1 is last evidence var
      r_prob += samples[key]
    w_probs.append(samples[key])
  w_prob = r_prob / sum(w_probs) # weighted probability of most likely sequence
  return w_prob
  

# cal_average_prob_and_var(probs)
#   1. Calculate the average probability, and its variance.
#
#   Parameters:
#     1. probs: A sequence of calculated probabilities.
def cal_average_prob_and_var(probs):
  mean = np.mean(probs)
  var = np.var(probs) 
  return mean, var


def main():
  # uncomment for submission
  n = int(sys.argv[1])
  size = int(sys.argv[2])
  evid_seq = []
  for i in range(size):
      evid_seq.append(int(sys.argv[i+3]))

  # uncomment for testing
  # n = 1000
  # size = 10
  # evid_seq = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

  init_prob = [0.5, 0.5]
  T = np.matrix([[0.7, 0.3],[0.3, 0.7]])
  E = np.matrix([[0.9, 0.1],[0.2, 0.8]])

  evid_seq = [abs(x - 1) for x in evid_seq]
  probs = []
  for i in range(10):
    samples = generate_samples(init_prob, evid_seq, T, E, n, size)
    probs.append(cal_weighted_prob_from_samples(samples))
  
  prob_mean, var = cal_average_prob_and_var(probs)

  print("Estimated probability: ", prob_mean)
  print("Variance of the estimation: ", var)




if __name__ == "__main__":
  main()