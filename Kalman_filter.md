# Filters and Kalman Filter

### What are filters?
Filters are a kind of network models that incorporate the **certainty(our knowledge) and uncertainty(the noise in real world)** of our **belief and observation** for a dynamic system in a sequence of time steps. 

### What are filters used for?
For a dynamic system, if we are 100% confident about our knowledge, we can simply predict the state in any time step. Or if we are 100% confident about the observations, we can simply calculate the system state in any time step based on observations. 

But the real world is complex, we usually don't have full knowledge of the system and the observations usually contain certain amount of noise. So we need a way to incorporate our knowledge and observation in all time steps as much as possible. This is where we use filters.

### How do filters work?
The generic framework of a filter follows these steps:

1. **Guess** a initial system state. Because we are not 100% confident about our guess, we use a probability distribution to represent our belief;
2. **Receive** the observation at this time step --again this observation is uncertain, we use probability distribution to represent it-- and **combine** (*take a value between*) the information in our prediction and observation and update the system state at this time step(we are gaining information coming from observation);
3. **Guess** the state in next time step using our knowledge of this system. Because we are uncertain about our knowledge of this system, the uncertainty adds up. In other words, we are losing the confidence or information;
4. **Repeat** step 2-3 for following time steps.

**The essence of filter is the combination of prediction and measurement, which is a weighted average of these 2 values.** If we are more confident about our prediction, then the new value will be closer to our prediction value. If we are more confident about observation, then the new value bias toward observed value. 

![](/Users/LAI/Documents/data_science/projects/wip/taxi_trip_pattern_learning/_documents/imgs/network.png)


### Common variables names used in literature:

- $x_t$: actual state value at time t  
- $\bar{x}_t$: state prior probability distribution at time t  
- $\hat{x}_t$: state posterior probability distribution at time t   


- $z_t$: actual observed value at time t  
- $\bar{z}_t$: prior probability distribution of observed variable predicted from $\bar{x}_t$  
- $\hat{z}_t$: posterior probability distribution of observed variable given $z_t$


- $P_t$: state variance, which is increasing in prediction step and decreaasing in update step. 
    + $\bar{P}_t$: the prior state variance
    + $\hat{P}_t$: the posterior state variance
- $Q$: process noise, part of transition model, which typically won't change.  
- $R$: measure noise, part of sensor model, which typically won't change.  


- $F$: transition model
- $H$: sensor model


### From a probabilistic point of view:

1. Guess the prior probability distribution of system state at $t_0$: 

    $$P(\bar{x}_0)$$   

2. Receive observation $prob(z_0)$ at $t_0$ and combine this observation as a posterior probability distribution with our guess using Bayesian theorem:  

    $$
    \begin{aligned}
        P(\hat{x}_0) 
        & = P(x_0|z_0)\\
        & = \frac{P(z_0|\bar{x}_0)P(\bar{x}_0)}{P(z_0)}\\
        & = \frac{P(z_0|\bar{x}_0)P(\bar{x}_0)}
            {\sum_{\bar{x}_0} P(z_0, \bar{x}_0)}
    \end{aligned}
    $$


3. Guess the prior probability distribution of system state at $t_1$: 

    $$P(\bar{x}_1) = \sum_{\hat{x}_0}(P(\bar{x}_1|\hat{x}_0)P(\hat{x}_0))$$
    
4. Repeat step 2-3 for following time steps.

Note that the conditional probability $P(z_0|\bar{x}_0)$ contains the knowledge of how system state generate observations(sensor model). It includes both situations when the observations are directly measurement of system state and when they are not(they are actually measurements of a related but different state).

And the conditional probability $P(\bar{x}_t|\hat{x}_{t-1})$ contains the knowledge of how system state evolve to next state(transition model).





### *Addition and Multiplication of probability distributions:
  
Both addtion and multiplication only fit the case when random variables are continuous.

- **Addition**:[Wikipedia](https://en.wikipedia.org/wiki/Convolution_of_probability_distributions)

    It means when 2 values adding up($Z = X + Y$), if X has probability distribution $P(X)$ and Y has probability distribution $P(Y)$, then Z has probability distribution $P(Z)$ which is the convolution of $P(X)$ and $P(Y)$.
    $$
        P(Z=z) = \int_{-\infty}^{\infty} P(X = x)P(Y = z-x) dx
    $$
    
- **Multiplicaiton**: [Wikipedia](https://en.wikipedia.org/wiki/Product_distribution)   

    Similar, if X and Y are two independent, continuous random variables, described by probability density functions $P(X)$ and $P(Y)$ then the probability density function of $Z = XY$ is:
    $$
        P(Z=z) = \int_{-\infty}^{\infty} P(X = x) P(Y = \frac{z}{x}) \frac{1}{\lvert x \rvert} dx
    $$

#### Addtion and Multiplication of Gaussian distributions
Fortunately, the addition and multiplicaiton of Gaussian distributions are quite easy:

- **Addition**:
    $$
        \mathcal{N}(\mu_1, \sigma_1^2) + \mathcal{N}(\mu_2, \sigma_2^2)
        = \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)
    $$
    
    
- **Linear transformation**:    
    We can treat this as addtion multiple times.
    $$
        a\mathcal{N}(\mu, \sigma^2) + b 
        = \mathcal{N}(a\mu + b, a^2\sigma^2)
    $$

- **Multiplication**:
    $$
        \mathcal{N}(\mu_1, \sigma_1^2) * \mathcal{N}(\mu_2, \sigma_2^2)
        = \mathcal{N}(
            \frac{\sigma_2^2\mu_1 + \sigma_1^2\mu_2}{\sigma_1^2 + \sigma_2^2}, 
            \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2 + \sigma_2^2}
    )$$


## What is Kalman filter?

Kalman filter is a special case of filters which parameterizes the previous probability distribution using Gaussian distributions. And we assume next state is a linear transformation of previous state add Gaussian noise: 

$$x_{t+1} = Fx_t + w,    w \sim \mathcal{N}(0, Q)$$ 
equivalently 
$$P(x_{t+1}|x_t) = \mathcal{N}(Fx_t, Q)$$


### In cases where observations are directly measures of system state:
We have observed variable:
$$z_t = x_t + v,   v \sim \mathcal{N}(0, R)$$
equivalently 
$$P(z_t|x_t) = \mathcal{N}(x_t, R)$$

**Prior State**:
$$
    P(\bar{x}_t) \sim \mathcal{N}(\bar{x}_t, \bar{P}_t)
$$  

**FORWARD Update**:
$$\begin{aligned}
    P(\hat{x}_t) \sim \mathcal{N}(\hat{x}_t, \hat{P}_t) 
    & = P(z|\bar{x}_t, R) \mathcal{N}(\bar{x}_t, \bar{P}_t) 
        && \text{here the likelihood use sensor noise} \\
    & = P(\bar{x}_t|z, R) \mathcal{N}(\bar{x}_t, \bar{P}_t) 
        && \text{because} d(\bar{x}_t,z_t) = d(z_t, \bar{x}_t) \\
    & = \mathcal{N}(z_t, R) \mathcal{N}(\bar{x}_t, \bar{P}_t) \\
    & = \mathcal{N}(
        \frac{\bar{P}_t z_t + R \bar{x}_t}{\bar{P}_t + R}, 
        \frac{\bar{P}_t R}{\bar{P}_t + R}) \\
    & = \mathcal{N}(
        \frac{\bar{P}_t}{\bar{P}_t + R} z_t + \frac{R}{\bar{P}_t + R}\bar{x}_t, 
        \frac{\bar{P}_t R}{\bar{P}_t + R}) \\
    & = \mathcal{N}(K z_t + (I-K)\bar{x}_t, KR) \\
    & = \mathcal{N}(\bar{x}_t + K(z_t - \bar{x}_t), KR) \\
\end{aligned}$$

where
$$
    K = \frac{\bar{P}_t}{\bar{P}_t + R}
$$

**FORWARD Predict**:
$$\begin{aligned}
    P(\bar{x}_{t+1}) \sim \mathcal{N}(\bar{x}_{t+1}, \bar{P}_{t+1}) 
    & = \int_{\hat{x}_t} P(\bar{x}_{t+1}|\hat{x}_t) P(\hat{x}_t) d\hat{x}_t\\
    & = \int_{\hat{x}_t} P(\bar{x}_{t+1}|F\hat{x}_t) P(F\hat{x}_t) d\hat{x}_t && \text{after linear transformation}\\
    & = \int_{\hat{x}_t} \mathcal{N}(\bar{x}_{t+1}; F x'_t, Q) \mathcal{N}(F x'_t; F\hat{x}_t, F\hat{P}_tF^\intercal ) d\hat{x}_t && \text{where x_t' is a value at time t}\\
    & = \int_{\hat{x}_t} \mathcal{N}(\bar{x}_{t+1} - F x'_t; 0, Q) \mathcal{N}(F x'_t; F\hat{x}_t, F \hat{P}_t F^\intercal) d\hat{x}_t && \text{move first Gaussian to original to match the convolution equation}\\
    & = \mathcal{N}(\bar{x}_{t+1}; F\hat{x}_t, F \hat{P}_t F^\intercal + Q)&& \text{and the convolution of 2 Gaussians is the addtion of them}\\
\end{aligned}$$


### In cases where observations are NOT directly measured on system state:

We assume observation $z_t$ is a linear transformation of state $x_t$: $z_t = Hx_t$. So we need to adjust the forward update function:

**The prior distribution of observed variable** $\bar{z}_t$:

$$
    P(\bar{z}_t) = \mathcal{N}(H\bar{x}_t, H\bar{P}_tH^\intercal)
$$


And the **likelihood**:
$$\begin{aligned}
    P(z_t|\bar{z}_t) 
    & = \mathcal{N}(z_t; \bar{z}_t, R)\\
    & = \mathcal{N}(z_t; H\bar{x}_t, R)\\
    & = \mathcal{N}(H\bar{x}_t; z_t, R)\\
    & = \mathcal{N}(z_t, R)
\end{aligned}$$


Thus we can calculate the **posterior observed variable distribution**:

$$\begin{aligned}
    P(\hat{z}_t)
    & = P(\bar{z}_t) P(z_t|\bar{z}_t)\\
    & = \mathcal{N}(H\bar{x}_t, H \bar{P}_t H^\intercal) \mathcal{N}(z_t, R)\\
    & = \mathcal{N}(
            \frac{R H\bar{x}_t + H \bar{P}_t H^\intercal z_t}{H \bar{P}_t H^\intercal + R}, 
            \frac{H \bar{P}_t H^\intercal R}{H \bar{P}_t H^\intercal + R}
        )
\end{aligned}$$


Because $P(\hat{z}_t) \sim \mathcal{N}(H\hat{x}_t, H\hat{P}_tH^\intercal)$,
$$
    H\hat{x}_t = \frac{R H \bar{x}_t + H \bar{P}_t H^\intercal z_t}{H \bar{P}_t H^\intercal + R}
$$
and
$$
    H \hat{P}_t H^\intercal = \frac{H \bar{P}_t H^\intercal R}{H \bar{P}_t H^\intercal + R}
$$

Solve for $\hat{x}_t$ and $\hat{P}_t$:
$$
    \hat{x}_t = \frac{R \bar{x}_t + \bar{P}_tH^\intercal z_t}{H\bar{P}_tH^\intercal + R}\\
    \hat{P}_t = \frac{\bar{P}_t R}{H\bar{P}_tH^\intercal + R}
$$

set
$$
    K_t = \frac{\bar{P}_tH^\intercal}{H\bar{P}_tH^\intercal + R} 
$$

we can rewrite $\hat{x}_t$ and $\hat{P}_t$:
$$\begin{aligned}
    \hat{x}_t 
    & = K_t z_t + (1 - HK_t)\bar{x}_t \\
    & = \bar{x}_t + K_t (z_t - H\bar{x}_t) \\
    \hat{P}_t
    & = \bar{P}_t \frac{R}{H\bar{P}_tH^\intercal + R} \\
    & = (I - K_tH) \bar{P}_t
\end{aligned}$$


### Several things to be noted:
- the posterior observation is the weighted average of prediction variance and measurement variance:

$$
    \frac{HPH^\intercal}{HPH^\intercal + R} \text{(real observation)} + \frac{R}{HPH^\intercal + R} \text{(prior observation)}
$$

- the posterior variance is independent of either predicted value or observed value, it only depends on $R$ and $\bar{P}_t$. So it can be computed before receiving the measurement.

### Parameter Learning in Kalman filter:
We can use the observed values from time 0 to time T to estimate the paramter of filter. This is a unsupervise learning problem. We can use EM algorithm to solve it:


#### E step:  
When we clamp the observation Z, and the paramters $\bar{x}_0, \bar{P}_0, F, H, Q, R$, this filter becomes a distribution of hidden variable X from time 0 to T. We can map this distribution to a scalar by taking the expectation value:

$$
    \text{Expected Likelihood} = E[P(X, Z|Z)]
$$

We can use the forward-backward algorithm to calcualte the likelihood--the probability of states given observations and parameters. And because maximizing the expected log likelihood equals to maximizing the expected likelihood but easier to solve in the M step, so we look at the log version instead:

$$
    \text{Expected Log Likelihood} = E[log P(X, Z|Z)]
$$

Let 
$$
    E_t^\tau \equiv E[\hat{x}_t|z_{1:\tau}]\\
    V_t^\tau \equiv Var[\hat{x}_t|z_{1:\tau}] = E[(x_t - \hat{x}_t)(x_t - \hat{x}_t)^\intercal|z_{1:\tau}]
$$

As we will see shortly in M step, maximizing the ELL needs the following quantities calculated for every time step:


$$\begin{aligned}
    E_t^T & = E[\hat{x}_t|Z_{1:T}]\\
    S_t^T & = E[\hat{x}_t\hat{x}_t^\intercal|Z_{1:T}]\\
    S_{t, t-1}^T 
    & = E[\hat{x}_t \hat{x}_{t-1}^T|Z_{1:T}]\\
    & = E[\hat{x}_{t-1} \hat{x}_t^T|Z_{1:T}]\\
    & = S_{t-1, t}^T\\
\end{aligned}$$


**Forward message**:
$$\begin{aligned}
    E_t^{t-1} & = F E_{t-1}^{t-1} && (1) && \text{predict: forward message from } E_{t-1}^{t-1} \text{ to } E_t^{t-1}\\
    V_t^{t-1} & = F V_{t-1}^{t-1} F^\intercal + Q && (2) && \text{predict: forward message from } V_{t-1}^{t-1} \text{ to } V_t^{t-1}\\
    K_{t} & = V_{t}^{t-1} H^\intercal (H V_{t}^{t-1} H^\intercal + R)^{-1} && (3) && \text{Kalman gain}\\
    E_t^t & = E_t^{t-1} + K_t (z_t - H E_t^{t-1}) && (4) && \text{update: forward message from } E_t^{t-1} \text{ to } E_t^t\\
    V_t^t & = V_t^{t-1} - K_t H V_{t}^{t-1} && (5) && \text{update: forward message from } V_t^{t-1} \text{ to } V_t^t\\
\end{aligned}$$

where $E_1^0 = \bar{x}_0$ and $V_1^0 = \bar{P}_0$


**Backward message**:
$$\begin{aligned}
    J_{t-1} & = V_{t-1}^{t-1} F^\intercal (V_{t}^{t-1})^{-1} && (6) && \text{backward Kalman gain}\\
    E_{t-1}^{T} & = E_{t-1}^{t-1} + J_{t-1} (E_t^T - F E_{t-1}^{t-1}) && (7) && \text{backward message from }E_t^T \text{ to } E_{t-1}^{T}\\
    V_{t-1}^{T} & = V_{t-1}^{t-1} + J_{t-1} (V_t^T - V_{t}^{t-1})J_{t-1}^\intercal && (8) && \text{backward message from }V_t^T \text{ to } V_{t-1}^{T}\\
    V_{t-1, t-2}^T & = V_{t-1}^(t-1) J_{t-2} ^\intercal + J_{t-1} (V_{t, t-1}^T - F V_{t-1}^{t-1}) J_{t-2} ^\intercal && (9) && \text{backward message from }V_{t, t-1}^T \text{ to } V_{t-1, t-2}^T\\
\end{aligned}$$


$$\begin{aligned}
    S_t^T & = V_t^T + E_t^T {E_t^T}^\intercal && (10)\\
    S_{t, t-1}^T & = V_{t, t-1}^T + E_t^T {E_{t-1}^T}^\intercal && (11)\\
\end{aligned}$$

where
$$\begin{aligned}
    V_{T, T-1}^T & = (I - K_T H) F V_{T-1}^{T-1} && (12)\\
\end{aligned}$$



**So for $E_t^T$**:  
We need to pass information forward using equation (1)(3) and (4) to calculate $E_t^t$ for each time step and pass information backward using equation (7) to calcualte $E_t^T$ for each time step.
    
**So for $S_t^T$**:  
Requires the information of $V_t^T$ and $E_t^T$. Like calculating $E_t^T$, we need to pass information forward using equation (3) and (5) to calculate $V_t^t$ for each time step and pass information backward using equation (8) to calcualte $V_t^T$ for each time step.
    
**So for $S_{t, t-1}^T$**:  
$S_{t, t-1}^T$ can be calculated passing information backward from time T using equation (11) and (9)


#### M step: 
Then we maximize the ELL treating the observation Z, and the hidden variable X as constant and paramters $\bar{x}_0=\pi, \bar{P}_0=V, F, H, Q, R$ as variable. The optimal values for these parameters can be obtained by setting the derivative of ELL w.r.t. each of them to 0 and solve the equation.

We denote the paramters after maximization as $parameter^{new}$.

**Sensor model H**:
$$\begin{aligned}
    \frac{\partial ELL}{\partial H} 
    & = - \sum_{t=0}^T R^{-1} z_t {E_t^T}^\intercal + \sum_{t=0}^T R^{-1} H V_t^T = 0\\
    H^{new} 
    & = \big( \sum_{t=0}^T z_t E_t^T \big) \big( \sum_{t=0}^T V_t^T \big)^{-1}\\
\end{aligned}$$


**Sensor noise variance R**:
$$\begin{aligned}
    \frac{\partial ELL}{\partial R^{-1}} 
    & = \frac{T}{2}R
    - \sum_{t=0}^T \big( \frac{1}{2} z_t z_t^\intercal - H E_t^T z_t^\intercal + \frac{1}{2}H V_t^T H^\intercal \big)= 0\\
    R^{new} & = \frac{1}{T} \sum_{t=0}^T (z_t z_t^\intercal - H^{new} E_t^T z_t^\intercal)\\ 
\end{aligned}$$


**transition model F**:
$$\begin{aligned}
    \frac{\partial ELL}{\partial F} 
    & = -\sum_{t=0}^{T-1} Q^{-1} V_{t, t-1}^T + \sum_{t=0}^{T-1} Q^{-1} F V_{t-1}^T = 0\\
    F^{new} & = \big( \sum_{t=0}^{T-1} V_{t, t-1}^T \big) \big( \sum_{t=0}^{T-1} V_{t-1}^T \big)^{-1}\\
\end{aligned}$$


**Transition noise covariance Q**:
$$\begin{aligned}
    \frac{\partial ELL}{\partial Q^{-1}} 
    & = \frac{T-1}{2}Q - \frac{1}{2} \sum_{t=0}^{T-1} (V_t^T - F V_{t-1, t}^T - V_{t, t-1}^T F^\intercal + F V_{t-1}^T F^\intercal)\\
    & = \frac{T-1}{2}Q - \frac{1}{2} \big( \sum_{t=0}^{T-1} V_t^T - F^{new} \sum_{t=0}^{T-1} V_{t-1, t}^T \big)
    = 0\\
    Q^{new} & = \frac{1}{T-1} \big( \sum_{t=0}^{T-1} V_t^T - F^{new} \sum_{t=0}^{T-1} V_{t-1, t}^T \big)\\
\end{aligned}$$


**Initial state mean $\pi$**:
$$\begin{aligned}
    \frac{\partial ELL}{\partial \bar{x}_0} 
    & = (E_0^T - \bar{x}_0) \bar{P}_0^{-1}
    = 0\\
    \bar{x}_0^{new} & = E_0^T\\
\end{aligned}$$


**Initial state covariance V**:
$$\begin{aligned}
    \frac{\partial ELL}{\partial \bar{P}_0^{-1}} 
    & = \frac{1}{2} \bar{P}_0 - \frac{1}{2}(V_0^T - E_0^T \bar{x}_0^\intercal - \bar{x}_0 {E_0^T}^\intercal + \bar{x}_0 \bar{x}_0^\intercal )
    = 0\\
    \bar{P}_0^{new} & = P_0^T - E_0^T {E_0^T}^\intercal\\
\end{aligned}$$


#### Reference:
- Zoubin Ghahramani and Geoffrey, E. Hinton.(1996)  Paramter Estimation for Linear Dynamical Systems.

$$\begin{aligned}
    \text{Expected Likelihood} 
    & = E[P(X, Z|Z)]\\
    & = E[P(\hat{x}_0)\prod_{t=0}^T P(z_t|\bar{x}_t) \prod_{t=0}^T P(\bar{x}_{t+1}|\hat{x}_t)]
\end{aligned}$$

And because maximize expected log likelihood equals to maxize expected likelihood but easier to solve,

$$\begin{aligned}
    \text{Expected Log Likelihood} 
    & = E\big[log \big( P(\hat{x}_0)\prod_{t=0}^T P(z_t|\bar{x}_t) \prod_{t=0}^T P(\bar{x}_{t+1}|\hat{x}_t) \big)\big]\\
    & = E\big[log(P(\hat{x}_0)) + log\big( \sum_{t=0}^T P(z_t|\bar{x}_t) \big) + log\big( \sum_{t=0}^T P(\bar{x}_{t+1}|\hat{x}_t) \big)\big]\\
    & = 
\end{aligned}$$

As we have seen, the intial state variable, the probability representation of transition model and sensor model are:
$$\begin{aligned}
    P(\hat{x}_0)
    & = \mathcal{N}(\hat{x}_0; \bar{x}_0, \bar{P}_0)\\
    & = exp\{-\frac{1}{2}[\hat{x}_0 - \bar{x}_0] \bar{P}_0 [\hat{x}_0 - \bar{x}_0]^T\} (2\pi)^{-k/2}\lvert\bar{P}_0\rvert^{-1/2}\\
    P(\bar{x}_{t+1}|\hat{x}_t) & = \mathcal{N}(\bar{x}_{t+1}; F\hat{x}_t, Q)\\
    P(z_t|\bar{x}_t) & = \mathcal{N}(z_t; H\bar{x}_t, R)\\
\end{aligned}$$


























