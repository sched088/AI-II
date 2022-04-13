# Problem 2 - Part (b)

import sys
import numpy as np
import random as rand

# generate_samples(init_prob, evid_seq, T, E, n)
#   1. Randomly generate a number of samples.
#
#   Parameters:
#     1. init_prob: Initial probabilities
#     2. evid_seq: Evidence seqeuence
#     3. T: T matrix
#     4. E: E matrix
#     5. n: Number of samples to be generated
#     6. size: Number of states in a sample
def generate_samples(init_prob, evid_seq, T, E, n, size):
  # initialize samples: R0
  samples = np.random.random(n)
  particles = []
  for sample in samples: # convert samples to rain, no_rain based on init_prob
    if sample < init_prob[0]: # 0.5 chance of rain
      particles.append(0) # rain
    else:
      particles.append(1) # no rain
 
  # propagate particles R1-Rsize
  i = 0
  while i < size:
    prop_parts = []
    prop_probs = np.random.random(n)

    for _ in range(len(particles)):
      if prop_probs[_] >= T[0,0]: # 0.3 chance of changing ## note: shortcut here because of equal probs of given T
        prop_parts.append(abs(particles[_]-1)) # update particle if it changes. Otherwise keep same
      else:
        prop_parts.append(particles[_])
  
    # weight samples
    weight_parts = []
    for _ in range(len(prop_parts)):
      weight_parts.append(E[prop_parts[_], evid_seq[i]])  
    resample = rand.choices((prop_parts), weights=weight_parts, k=n)
    particles = resample
    
    i+=1
  particles = [abs(x - 1) for x in particles]
  # print(particles)
  return particles
        

# cal_weighted_prob_from_samples(samples)
#   1. Calculate the probability, based on the samples.
#
#   Parameters:
#     1. samples: Generated samples.
""" note: here we are back to original formatting where 0 = F and 1 = T
    each set of particles is the sampling at R10. Where it is 1 we are predicting T weighted by the evidence
"""
def cal_weighted_prob_from_samples(samples):
  r_prob = 0
  for _ in samples:
    if _ == 1:
      r_prob += 1
  w_prob = r_prob / len(samples)
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
  # n = 1000 # number of samples/particles
  # size = 10 # steps to take along evidence sequence
  # evid_seq = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

  init_prob = [0.5, 0.5]
  T = np.matrix([[0.7, 0.3],[0.3, 0.7]])
  E = np.matrix([[0.9, 0.1],[0.2, 0.8]])
  evid_seq = [abs(x - 1) for x in evid_seq] # convert F = 0 to F = 1 and from T = 1 to T = 0 to align with matrices

  probs = []
  for i in range(10):
    samples = generate_samples(init_prob, evid_seq, T, E, n, size)
    probs.append(cal_weighted_prob_from_samples(samples))
  
  prob_mean, var = cal_average_prob_and_var(probs)

  print("Estimated probability: ", prob_mean)
  print("Variance of the estimation: ", var)




if __name__ == "__main__":
  main()

'''Code that I tried that didn't work'''

    # # I think this would work properly with normalization. 
    # resample = []
    # w_samples = np.random.uniform(0,max(weight_parts),n)
    # # print(w_samples)
    # for _ in range(len(weight_parts)):
    #   if w_samples[_] <= weight_parts[_]:
    #     resample.append(prop_parts[_])
    #   else:
    #     resample.append(abs(prop_parts[_]-1))

'''more: again if I had weighted these prior I think it could have worked.'''

# weight_parts = []
#     if evid_seq[i] == 0: # if there is evidence of umbrella
#       for particle in prop_parts:
#         weight = np.random.random()
#         if particle == 0: # if particle sampled rain then 90% chance of umbrella
#           if weight > E[0,0]:
#             weight_parts.append(1)
#           else:
#             weight_parts.append(0)
#         if particle == 1: # partricle sampled no_rain then 20% chance of umbrella
#           if weight > E[1,0]:
#             weight_parts.append(0)
#           else:
#             weight_parts.append(1)
#     else: # evidence of no_umbrella
#       for particle in particles:
#         weight = np.random.random()
#         if particle == 0: # if particle sampled rain then 10% chance of no_umbrella
#           if weight > E[0,1]:
#             weight_parts.append(1)
#           else:
#             weight_parts.append(0)
#         if particle == 1: # particle sampled no_rain then 80% chance of no_umbrella
#           if weight > E[1,1]:
#             weight_parts.append(0)
#           else:
#             weight_parts.append(1)
#     print('weigh_parts')
#     print(weight_parts)
#     print('no_rain: ' + str(weight_parts.count(1)), 'rain:' + str(weight_parts.count(0)))
#     # resample
#     resample = []
#     j = 0
#     while j < n:
#       new_sample = np.random.randint(0,n)
#       resample.append(weight_parts[new_sample])
#       j+=1
#     print(resample)
#     particles = resample

#     print('resample')
#     print('no_rain: ' + str(resample.count(1)), 'rain:' + str(resample.count(0)))