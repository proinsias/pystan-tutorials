#!/usr/bin/env python
# -*- coding: utf-8 -*-
# From pystan quick start [guide](https://pystan.readthedocs.io/en/latest/getting_started.html).
# In[1]:
import arviz
import pystan
# In[2]:
get_ipython().run_line_magic('matplotlib', 'inline')
# In[3]:
sm = pystan.StanModel(file='../stan/8schools.stan')
# In[4]:
schools_dat = {
    'J': 8,
    'y': [28,  8, -3,  7, -1,  1, 18, 12],
    'sigma': [15, 10, 16, 11,  9, 11, 10, 18],
}
# In[5]:
fit = sm.sampling(data=schools_dat, iter=1000, chains=4)
# In[6]:
la = fit.extract(permuted=True)  # return a dictionary of arrays
mu = la['mu']
## return an array of three dimensions: iterations, chains, parameters
a = fit.extract(permuted=False)
# In[7]:
print(fit)
# In[9]:
arviz.plot_trace(fit);
# In[ ]:
