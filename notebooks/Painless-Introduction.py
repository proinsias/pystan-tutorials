#!/usr/bin/env python
# coding: utf-8
# [Code](https://gist.github.com/sergiosonline/6a3e0b1345c8f002d0e7b11aaf252d44)
# for
# [Painless Introduction to Applied Bayesian Inference using (Py)Stan](https://towardsdatascience.com/painless-introduction-to-applied-bayesian-inference-using-py-stan-36b503a4cd80).
# In[1]:
import pystan
import pickle
import numpy as np
import arviz as az
import pandas as pd
import seaborn as sns
import statsmodels.api as statmod
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.core.display import HTML 
# # Bayesian Inference - Estimate $\theta$ from Bernoulli observations
# $$
# \theta \sim \text{beta}(a,b)\\
# Y \sim \text{bernoulli}(\theta)
# $$
# 
# $$
# P(Y=y|\theta) = \theta^{y} (1-\theta)^{1-y},\\ \text{where } Y \in \{0,1\}, \theta \in [0,1]
# $$
# 
# $$
# P(\theta|Y) \approx \frac{\prod^{K}P(Y=y_{i}|\theta) P(\theta)}{C}
# $$
# 
# Recall that $y \sim \text{Bernoulli}(\theta)$ is equivalent to
# $$
# P(Y=y|\theta) = \theta^{y} (1-\theta)^{1-y},\\ \text{where } Y \in \{0,1\}, \theta \in [0,1]
# $$
# 
# Imagine we have an arbitrary coin and we would like to determine whether it is fair ($\theta$ = 50%) or not. For this problem, we observe a sample of Y's (1 indicates heads, while 0 indicates tails).
# 
# We experiment by flipping this coin K number of times. We would like to estimate $\theta$, i.e., the probability that we obtain heads for a given toss.
# 
# If we tackle this problem from the frequentist perspective, we can easily devise the maximum likelihood estimator for $\theta$ as 
# $$\hat{\theta}_{ML} = \sum_{i}^{K}\frac{y_{i}}{K},$$ where $K$ corresponds to the number of trials/observations in our experiment.
# 
# Say we flip the coin **4** times and, by struck of luck (or lack thereof), we observe **all** tails. Then, 
# 
# $$\hat{\theta}_{ML} = \sum_{i}^{4}\frac{y_{i}}{4}=0$$
# 
# which is quite extreme, for, having two sides, there is **SOME** probability that we can observe heads in the next trial.
# #### Applied Bayesian inference
# 
# We can adopt a more 'scientific' approach by expressing our prior belief about this coin. Say I believe the coin is most likely to be fair, but I believe there to be a (possibly smaller) chance that it is not.
# 
# Then I can establish the following prior probability on $\theta$:
# $$
# \theta \sim \text{beta}(5,5)
# $$
# 
# This prior makes sense because $\forall \alpha, \beta \in \mathcal{R}^+, \text{ beta}(\alpha,\beta) \in [0,1]$, namely, $\theta$ is a probability and **cannot** be less than 0 or greater than 1.
# 
# Let's visualize this prior by sampling from it:
# In[2]:
sns.distplot(np.random.beta(5,5, size=10000),kde=False);
# In[3]:
# bernoulli model
data = dict(N=4, y=[0, 0, 0, 0])
model_code = """
    data {
      int<lower=0> N;
      int<lower=0,upper=1> y[N];
    }
    parameters {
      real<lower=0,upper=1> theta;
    }
    model {
      theta ~ beta(5, 5);
      for (n in 1:N)
          y[n] ~ bernoulli(theta);
    }
    """
# In[4]:
model = pystan.StanModel(model_code=model_code)
# In[5]:
fit = model.sampling(data=data,iter=4000, chains=4, warmup=1000)
# Default params for sampling() are iter=1000, chains=4, warmup=500
# In[6]:
la = fit.extract(permuted=True)  # return a dictionary of arrays
# In[7]:
print(fit.stansummary())
# In[8]:
ax = az.plot_trace(fit, var_names=["theta"])
# In[9]:
sns.distplot(np.random.beta(5,5, size=10000),kde=False);
# $$
# \begin{align}
# P(\theta|D)&=\frac{P(D|\theta)P(\theta)}{P(D)} \\
#     &\propto P(D|\theta)P(\theta)
# \end{align}
# $$
# 
# $$
# P(D) = \int P(D|\theta)P(\theta)d\theta
# $$
# # Bayesian Regression - Car MPG Problem
# Our dataset comes from [here](https://archive.ics.uci.edu/ml/datasets/auto+mpg)
# 
# Predict/estimate `mpg` based on available features:
# * `weight`
# * `year`
# * `cylinders`
# * `hp`
# * `acceleration`
# * `displacement`
# * `origin`
# In[10]:
Image(url= "https://external-preview.redd.it/fCEkl9G2WE3sw6VHYU8TM1J5K7zgmvECNWv__srdML0.jpg?auto=webp&s=5b8973a18dea7d517229b6dca1123c354bbe7603")
# Let's start with the regression equation for it shall inform how we think of our models, both as a maximum-likelihood curve-fitting problem and Bayesian generative modelling problem:
# 
# $$
# Y = X\beta + \epsilon \\
# \text{where } \epsilon \sim \text{MVN}(0,\Sigma)
# $$
# 
# Note that the only random quantity in our equation above is $\epsilon$.
# 
# In[11]:
# Load the data
cars_data = pd.read_csv("../data/cars.csv.gz")
print(cars_data.shape)
cars_data.head()
# In[12]:
for col_name in cars_data.columns[1:]:
    print(cars_data[col_name].value_counts().head())
# In[13]:
cars_data.loc[cars_data['origin']==1, 'origin'] = 'American'
cars_data.loc[cars_data['origin']==2, 'origin'] = 'European'
cars_data.loc[cars_data['origin']==3, 'origin'] = 'Japanese'
# In[14]:
sns.distplot(cars_data[cars_data['origin']=='American']['mpg'],color="skyblue", label="American",kde=False)
sns.distplot(cars_data[cars_data['origin']=='Japanese']['mpg'],color="red", label="Japanese",kde=False)
sns.distplot(cars_data[cars_data['origin']=='European']['mpg'],color="yellow", label="European",kde=False)
plt.legend();
# In[15]:
f, axes = plt.subplots(2, 3, figsize=(7, 7), sharex=False)
sns.relplot(x="cylinders", y="mpg", data=cars_data, ax=axes[0, 0]);
sns.relplot(x="displacement", y="mpg", data=cars_data, ax=axes[0, 1]);
sns.relplot(x="horsepower", y="mpg", data=cars_data, ax=axes[0, 2]);
sns.relplot(x="acceleration", y="mpg", data=cars_data, ax=axes[1, 0]);
sns.relplot(x="model year", y="mpg", data=cars_data, ax=axes[1, 1]);
sns.relplot(x="weight", y="mpg", data=cars_data, ax=axes[1, 2]);
# close pesky empty plots
for num in range(2,8):
    plt.close(num);
    
plt.show();
# ### Maximum Likelihood OLS Regression
# In[16]:
from numpy import random
from sklearn import preprocessing, metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
random.seed(12345)
cars_data_vars = cars_data.columns.values[:-1].tolist()
# In[17]:
y = cars_data['mpg']
X = cars_data.loc[:, cars_data.columns != 'mpg']
X = X.loc[:, X.columns != 'name']
X = pd.get_dummies(X, prefix_sep='_', drop_first=False)
X = X.drop(columns=["origin_Japanese"])  # 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train.head()
# In[18]:
# Fit and summarize OLS model
X_train['const'] = 1; X_test['const'] = 1
#X_train = X_train.drop(columns=['origin_American'])
mod1 = statmod.OLS(y_train, X_train, prepend=False)
res = mod1.fit()
print(res.summary())
# In[19]:
y_train_freq = res.predict(X_train)
y_train_freq.head()
# In[20]:
y_test_freq = res.predict(X_test)
y_test_freq.head()
# In[21]:
freq_train_mse = metrics.mean_squared_error(y_train, y_train_freq)
print('Freq Train MSE:', freq_train_mse)
dff = pd.DataFrame({'y_pred':y_train_freq, 'y_obs':y_train})
grid = sns.JointGrid(dff.y_pred, dff.y_obs, space=0, height=6, ratio=50,
                    xlim=(0,50), ylim=(0,50))
grid.plot_joint(plt.scatter, color="b")
x0, x1 = grid.ax_joint.get_xlim()
y0, y1 = grid.ax_joint.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
grid.ax_joint.plot(lims, lims, ':k')
plt.subplots_adjust(top=0.9)
grid.fig.suptitle('Freq Train Predicted vs Observed',fontsize=20)
plt.show()
# In[22]:
freq_test_mse = metrics.mean_squared_error(y_test, y_test_freq)
print('Freq Test MSE:', freq_test_mse)
dff = pd.DataFrame({'y_pred':y_test_freq, 'y_obs':y_test})
grid = sns.JointGrid(dff.y_pred, dff.y_obs, space=0, height=6, ratio=50,
                    xlim=(0,50), ylim=(0,50))
grid.plot_joint(plt.scatter, color="b")
x0, x1 = grid.ax_joint.get_xlim()
y0, y1 = grid.ax_joint.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
grid.ax_joint.plot(lims, lims, ':k')
plt.subplots_adjust(top=0.9)
grid.fig.suptitle('Freq Test Predicted vs Observed',fontsize=20)
plt.show()
# ### Bayesian OLS Regression
# Now let's examine the Bayesian formulation of our regression problem:
# 
# $$
# Y = X\beta + \epsilon \\
# \text{where } \epsilon \sim \text{MVN}(0,\Sigma) \\
# \beta \sim P \\
# \sigma \sim Q
# $$
# 
# Recall the expression to evaluate the Bayesian posterior of the model parameters:
# 
# $$
# P(\theta|D) \propto P(D|\theta)P(\theta)
# $$
# 
# For our inference we are interested in obtaining posterior distributions for $\beta$ and $\sigma$.
# 
# Once we obtain an approximation of the posterior $P(\theta|D)$, we can actually compute a predictive posterior to fit an unseen/test vector of data points $X_{\text{new}}$:
# 
# $$
# P(X_{new}|D_{old}) = \int{P(X_{new}|\theta,D_{old})P(\theta|D_{old})}\,d\theta
# $$
# 
# In[23]:
from numpy import random
from sklearn import preprocessing, metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
random.seed(12345)
y = cars_data['mpg']
X = cars_data.loc[:, cars_data.columns != 'mpg']
X = X.loc[:, X.columns != 'name']
X = pd.get_dummies(X, prefix_sep='_', drop_first=False)
X = X.drop(columns=["origin_European"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
# In[24]:
X_train.head()
# In[25]:
# Succint matrix notation
cars_code = """
data {
    int<lower=1> N; // number of training samples
    int<lower=0> K; // number of predictors
    matrix[N, K] x; // matrix of predictors
    vector[N] y_obs; // observed/training mpg
    
    int<lower=1> N_new;
    matrix[N_new, K] x_new;
}
parameters {
    real alpha;
    vector[K] beta;
    //vector[K] tau;
    real<lower=0> sigma;
    
    vector[N_new] y_new;
}
transformed parameters {
    vector[N] theta;
    theta = alpha + x * beta;
}
model {
    sigma ~ exponential(1);
    alpha ~ normal(0, 6);
    beta ~ multi_normal(rep_vector(0, K), diag_matrix(rep_vector(1, K)));
    y_obs ~ normal(theta, sigma);
    
    y_new ~ normal(alpha + x_new * beta, sigma); // prediction model
}
"""
# In[26]:
cars_dat = {'N': X_train.shape[0],
            'N_new': X_test.shape[0],
            'K': X_train.shape[1],
            'y_obs': y_train.values.tolist(),
            'x': np.array(X_train),
            'x_new': np.array(X_test)}
# In[27]:
sm = pystan.StanModel(model_code=cars_code, verbose=True)  # FIXME: SET VERBOSE TO FALSE.
# In[28]:
# fit = sm.sampling(data=cars_dat, iter=6000, chains=8, n_jobs=1)
# Took 14 hours.
# In[ ]:
fit = sm.sampling(data=cars_dat, iter=6000, chains=8)
# Took 4.5 hours.
# In[29]:
la = fit.extract(permuted=True)
print(fit.stansummary())
# In[30]:
post_data = az.convert_to_dataset(fit)
# In[31]:
# Let's examine the posterior distribution of our model parameters
axes = az.plot_forest(
    post_data,
    kind="forestplot",
    var_names= ["alpha","beta","sigma"],
    combined=True,
    ridgeplot_overlap=1.5,
    colors="blue",
    figsize=(9, 4),
)
# # Print out some MC diagnostic plots (trace/mixing plots, etc)
# In[32]:
ax = az.plot_trace(fit, var_names=["alpha","beta","sigma"])
# Great mixing: The thicker the better
# In[33]:
bay_train_mse = metrics.mean_squared_error(y_train.values, fit['theta'].mean(0))
print('Bayes Train MSE:', bay_train_mse)
dff = pd.DataFrame({'y_pred':fit['theta'].mean(0), 'y_obs':y_train.values})
grid = sns.JointGrid(dff.y_pred, dff.y_obs, space=0, height=6, ratio=50,
                    xlim=(0,50), ylim=(0,50))
grid.plot_joint(plt.scatter, color="g")
x0, x1 = grid.ax_joint.get_xlim()
y0, y1 = grid.ax_joint.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
grid.ax_joint.plot(lims, lims, ':k')
plt.subplots_adjust(top=0.9)
grid.fig.suptitle('Bayes Train Pred vs Obs',fontsize=20)
plt.show()
# In[34]:
bay_test_mse = metrics.mean_squared_error(y_test, la['y_new'].mean(0))
print('Bayes Test MSE:', bay_test_mse)
dff = pd.DataFrame({'y_pred':la['y_new'].mean(0), 'y_obs':y_test})
grid = sns.JointGrid(dff.y_pred, dff.y_obs, space=0, height=6, ratio=50,
                    xlim=(0,50), ylim=(0,50))
grid.plot_joint(plt.scatter, color="b")
x0, x1 = grid.ax_joint.get_xlim()
y0, y1 = grid.ax_joint.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
grid.ax_joint.plot(lims, lims, ':k')  
plt.subplots_adjust(top=0.9)
grid.fig.suptitle('Bayes Test Predicted vs Obs',fontsize=20)
plt.show()
# # Visualize the value of the prediction as we vary some model inputs, along with the prediction 95% confidence bands
# In[35]:
import matplotlib.pyplot as plt
import weakref
az.style.use("arviz-darkgrid")
sns.relplot(x="weight", y="mpg",
            data=pd.DataFrame({'weight':X_test['weight'],'mpg':y_test}))
az.plot_hpd(X_test['weight'], la['y_new'], color="k", plot_kwargs={"ls": "--"})
# In[36]:
sns.relplot(x="acceleration", y="mpg",
            data=pd.DataFrame({'acceleration':X_test['acceleration'],'mpg':y_test}))
az.plot_hpd(X_test['acceleration'], la['y_new'], color="k", plot_kwargs={"ls": "--"})
# In[37]:
print('Freq Train MSE:', freq_train_mse)
print('Freq Test MSE:', freq_test_mse)
print('Bayes Train MSE:', bay_train_mse)
print('Bayes Test MSE:', bay_test_mse)
# 
# <table>
#   <tr>
#     <th></th>
#     <th>Train MSE</th>
#     <th>Test MSE</th> 
#   </tr>
#   <tr>
#     <td><b>Bayesian</b></td>
#     <td>10.829</td> 
#     <td>10.968</td> 
#   </tr>
#   <tr>
#     <td><b>Frequentist / MLE</b></td>
#     <td>10.747</td> 
#     <td>10.558</td> 
# </table>
# 
