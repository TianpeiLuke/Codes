#utf-8

# Import relevant modules
import pymc
import example_model1
#import matplotlib.pyplot as plt
#import scipy.stats as stats
import scipy.io as scio

S = pymc.MCMC(example_model1, db='pickle')
S.sample(iter=10000, burn=5000, thin=2)
pymc.Matplot.plot(S)
beta_trace = S.trace('beta', chain=None)[:]

scio.savemat('beta.mat', {'beta': beta_trace})
