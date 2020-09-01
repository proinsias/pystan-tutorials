#!/usr/bin/env python
# coding: utf-8
# From [kaggle.com](https://www.kaggle.com/gkoundry/bayesian-logistic-regression-with-pystan?).
# In[28]:
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan
import seaborn as sns
# In[2]:
train = pd.read_csv('../data/train.csv.gz')                                                            
train.pop('id')                                                                                      
target = train.pop('target').astype(int)
# In[3]:
print(train.shape)
train.head()
# In[4]:
test = pd.read_csv('../data/test.csv.gz')
ids = test.pop('id') 
# In[5]:
print(test.shape)
test.head()
# In[6]:
data = {                                                                                             
    'N': 250,                                                                                        
    'N2': 19750,                                                                                     
    'K': 300,                                                                                        
    'y': target,                                                                                     
    'X': train,                                                                                      
    'new_X': test,                                                                                   
}                                                                                                    
# In[7]:
code = """                                                                                         
data {                                                                                               
  int N; //the number of training observations                                                       
  int N2; //the number of test observations                                                          
  int K; //the number of features                                                                    
  int y[N]; //the response                                                                           
  matrix[N,K] X; //the model matrix                                                                  
  matrix[N2,K] new_X; //the matrix for the predicted values                                          
}                                                                                                    
parameters {                                                                                         
  real alpha;                                                                                        
  vector[K] beta; //the regression parameters                                                        
}                                                                                                    
transformed parameters {                                                                             
  vector[N] linpred;                                                                                 
  linpred = alpha+X*beta;                                                                            
}                                                                                                    
model {                                                                                              
  alpha ~ cauchy(0,10); //prior for the intercept following Gelman 2008                              
                                                                                                     
  for(i in 1:K)                                                                                      
    beta[i] ~ student_t(1, 0, 0.03);                                                                 
                                                                                                     
  y ~ bernoulli_logit(linpred);                                                                      
}                                                                                                    
generated quantities {                                                                               
  vector[N2] y_pred;                                                                                 
  y_pred = alpha+new_X*beta; //the y values predicted by the model                                   
}                                                                                                    
"""               
# In[8]:
sm = pystan.StanModel(model_code=code, verbose=True, )
# In[9]:
fit = sm.sampling(data=data, seed=101, verbose=True, n_jobs=2)
# Changed seed, set n_jobs to 2.
# In[10]:
ex = fit.extract(permuted=True)
# In[11]:
print(fit)
# In[12]:
print(fit.stansummary())
# In[13]:
az.plot_trace(fit);
# In[14]:
target = np.mean(ex['y_pred'], axis=0)                                                               
# In[15]:
df = pd.DataFrame({'id': ids, 'target': target})
# In[16]:
print(df.shape)
df.head()
# In[20]:
post_data = az.convert_to_dataset(fit)
# In[23]:
def plot_trace(param, param_name='parameter'):
  """Plot the trace and posterior of a parameter."""
  
  # Summary statistics
  mean = np.mean(param)
  median = np.median(param)
  cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)
  
  # Plotting
  plt.subplot(2,1,1)
  plt.plot(param)
  plt.xlabel('samples')
  plt.ylabel(param_name)
  plt.axhline(mean, color='r', lw=2, linestyle='--')
  plt.axhline(median, color='c', lw=2, linestyle='--')
  plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
  plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
  plt.title('Trace and Posterior Distribution for {}'.format(param_name))
  plt.subplot(2,1,2)
  plt.hist(param, 30, density=True); sns.kdeplot(param, shade=True)
  plt.xlabel(param_name)
  plt.ylabel('density')
  plt.axvline(mean, color='r', lw=2, linestyle='--',label='mean')
  plt.axvline(median, color='c', lw=2, linestyle='--',label='median')
  plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
  plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)
  
  plt.gcf().tight_layout()
  plt.legend()
# In[24]:
summary_dict = fit.summary()
df = pd.DataFrame(summary_dict['summary'], 
                  columns=summary_dict['summary_colnames'], 
                  index=summary_dict['summary_rownames'])
df.head()
# In[26]:
# Extracting traces
alpha = fit['alpha']
beta1 = fit['beta[1]']
beta2 = fit['beta[2]']
beta3 = fit['beta[3]']
beta4 = fit['beta[4]']
# In[29]:
plot_trace(alpha, 'alpha') 
# In[31]:
plot_trace(beta1, 'beta1') 
# In[32]:
plot_trace(beta2, 'beta2') 
# In[33]:
plot_trace(beta3, 'beta3') 
# In[34]:
plot_trace(beta4, 'beta4') 
# In[37]:
# Let's examine the posterior distribution of our model parameters
axes = az.plot_forest(
    post_data,
    kind="forestplot",
    var_names= ["alpha","beta"],
    combined=True,
    ridgeplot_overlap=1.5,
    colors="blue",
    figsize=(9, 4),
)
