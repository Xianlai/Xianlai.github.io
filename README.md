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

- <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/23776aad854f2d33e83e4f4cad44e1b9.svg?invert_in_darkmode" align=middle width=14.307150000000002pt height=14.102549999999994pt/>: actual state value at time t  
- <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/54099e7f60e197f7ae7167ecd3b3496e.svg?invert_in_darkmode" align=middle width=14.307150000000002pt height=18.597149999999978pt/>: state prior probability distribution at time t  
- <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/b189f0bc978e4a7853de920920cd11f4.svg?invert_in_darkmode" align=middle width=14.307150000000002pt height=22.745910000000016pt/>: state posterior probability distribution at time t   


- <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/a33a6206486276bf957212fceb1ecf46.svg?invert_in_darkmode" align=middle width=12.563430000000002pt height=14.102549999999994pt/>: actual observed value at time t  
- <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/9e65cdd9a6c52245b28212e1df41174a.svg?invert_in_darkmode" align=middle width=12.563430000000002pt height=18.597149999999978pt/>: prior probability distribution of observed variable predicted from <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/54099e7f60e197f7ae7167ecd3b3496e.svg?invert_in_darkmode" align=middle width=14.307150000000002pt height=18.597149999999978pt/>  
- <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/7ce0eedffece99a0a19955c78df56c94.svg?invert_in_darkmode" align=middle width=12.563430000000002pt height=22.745910000000016pt/>: posterior probability distribution of observed variable given <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/a33a6206486276bf957212fceb1ecf46.svg?invert_in_darkmode" align=middle width=12.563430000000002pt height=14.102549999999994pt/>


- <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/1fbfa938f7f0ee3057f94c2c2b9dc712.svg?invert_in_darkmode" align=middle width=15.461490000000001pt height=22.381919999999983pt/>: state variance, which is increasing in prediction step and decreaasing in update step. 
    + $\bar{P}_t$: the prior state variance
    + $\hat{P}_t$: the posterior state variance
- <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg?invert_in_darkmode" align=middle width=12.946890000000003pt height=22.381919999999983pt/>: process noise, part of transition model, which typically won't change.  
- <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode" align=middle width=12.56145pt height=22.381919999999983pt/>: measure noise, part of sensor model, which typically won't change.  


- <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/b8bc815b5e9d5177af01fd4d3d3c2f10.svg?invert_in_darkmode" align=middle width=12.805980000000003pt height=22.381919999999983pt/>: transition model
- <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode" align=middle width=14.944050000000002pt height=22.381919999999983pt/>: sensor model


### From a probabilistic point of view:

1. Guess the prior probability distribution of system state at <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/6df6ddacc987bd7a5070beafef47fcc1.svg?invert_in_darkmode" align=middle width=12.441990000000002pt height=20.14650000000001pt/>: 

    $$P(\bar{x}_0)$$   

2. Receive observation <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/a22d3c1f73b27bfdd047a7a5981609d4.svg?invert_in_darkmode" align=middle width=40.525650000000006pt height=24.56552999999997pt/> at <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/6df6ddacc987bd7a5070beafef47fcc1.svg?invert_in_darkmode" align=middle width=12.441990000000002pt height=20.14650000000001pt/> and combine this observation as a posterior probability distribution with our guess using Bayesian theorem:  
    $$
    \begin{aligned}
        P(\hat{x}_0) 
        & = P(x_0|z_0)\\
        & = \frac{P(z_0|\bar{x}_0)P(\bar{x}_0)}{P(z_0)}\\
        & = \frac{P(z_0|\bar{x}_0)P(\bar{x}_0)}
            {\sum_{\bar{x}_0} P(z_0, \bar{x}_0)}
    \end{aligned}
    $$


3. Guess the prior probability distribution of system state at <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/4ad941990ade99427ec9730e46ddcdd4.svg?invert_in_darkmode" align=middle width=12.441990000000002pt height=20.14650000000001pt/>: 

    $$P(\bar{x}_1) = \sum_{\hat{x}_0}(P(\bar{x}_1|\hat{x}_0)P(\hat{x}_0))$$
    
4. Repeat step 2-3 for following time steps.

Note that the conditional probability <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/7ab0d1c8ec04dc60a6083d6e83baff2b.svg?invert_in_darkmode" align=middle width=61.844145000000005pt height=24.56552999999997pt/> contains the knowledge of how system state generate observations(sensor model). It includes both situations when the observations are directly measurement of system state and when they are not(they are actually measurements of a related but different state).

And the conditional probability <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/10f513cd363e6a121c423dda74358b9e.svg?invert_in_darkmode" align=middle width=77.2761pt height=24.56552999999997pt/> contains the knowledge of how system state evolve to next state(transition model).


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

Kalman filters are a special kind of filters which parameterize the previous probability distribution as Gaussian distributions: we assume next state is a linear transformation of previous state add Gaussian noise: 

<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/2a06e7d2f99a272b1e6cb96b72b8cd4c.svg?invert_in_darkmode" align=middle width=212.18009999999998pt height=16.376943pt/></p> 
equivalently 
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/be20fb8d7242ab104001ab3572480ddc.svg?invert_in_darkmode" align=middle width=175.68869999999998pt height=16.376943pt/></p>


### In cases where observations are directly measures of system state:
We have observed variable:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/2838132ae4bffe287db3bfcb6dc8237a.svg?invert_in_darkmode" align=middle width=173.316pt height=16.376943pt/></p>
equivalently 
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/a01c650f964dfae5107628e439aa72d9.svg?invert_in_darkmode" align=middle width=144.10076999999998pt height=16.376943pt/></p>

**Prior State**:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/dfe1fa3e30193683c4e3dc1f3d1c7b99.svg?invert_in_darkmode" align=middle width=129.99739499999998pt height=17.547915pt/></p>  

**FORWARD Update**:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/cd4925c894586952bb82fff3a092dc0e.svg?invert_in_darkmode" align=middle width=663.9418499999999pt height=210.55814999999998pt/></p>

where
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/999ee4ab6ee429ff10367d50ceebf77c.svg?invert_in_darkmode" align=middle width=87.968265pt height=38.72022pt/></p>


**FORWARD Predict**:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/7ad7ce48bdec8f109dda6d5a9d5ba10c.svg?invert_in_darkmode" align=middle width=2629.968pt height=130.13814pt/></p>


### In cases where observations are NOT directly measured on system state:

We assume observation <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/a33a6206486276bf957212fceb1ecf46.svg?invert_in_darkmode" align=middle width=12.563430000000002pt height=14.102549999999994pt/> is a linear transformation of state <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/23776aad854f2d33e83e4f4cad44e1b9.svg?invert_in_darkmode" align=middle width=14.307150000000002pt height=14.102549999999994pt/>: <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/3d6b178c8aac4ed1f80c7fdbfb205c93.svg?invert_in_darkmode" align=middle width=64.529685pt height=22.381919999999983pt/>. So we need to adjust the forward update function:

**The prior distribution of observed variable** <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/9e65cdd9a6c52245b28212e1df41174a.svg?invert_in_darkmode" align=middle width=12.563430000000002pt height=18.597149999999978pt/>:

<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/d0993a19fa15bb73cb2d76de6db522fb.svg?invert_in_darkmode" align=middle width=181.35809999999998pt height=17.547915pt/></p>


And the **likelihood**:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/0635ba25c9c4bcbaca09c883e1c8e278.svg?invert_in_darkmode" align=middle width=178.19504999999998pt height=90.34954499999999pt/></p>


Thus we can calculate the **posterior observed variable distribution**:

<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/97f72a0c09cc86b35b036db3f46b97a4.svg?invert_in_darkmode" align=middle width=322.92645pt height=88.18854pt/></p>


Because <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/c5a91c11a1ab24528f8f43e963a61c2f.svg?invert_in_darkmode" align=middle width=181.35859499999998pt height=31.056300000000004pt/>,
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/780ec43c2abf239643b09f854c53885d.svg?invert_in_darkmode" align=middle width=184.89405pt height=38.72022pt/></p>
and
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/26925bd8dd2998f7fbc47c3588cb0386.svg?invert_in_darkmode" align=middle width=165.65175pt height=38.72022pt/></p>

Solve for <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/b189f0bc978e4a7853de920920cd11f4.svg?invert_in_darkmode" align=middle width=14.307150000000002pt height=22.745910000000016pt/> and <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/3166354ea06dfc8f7db7e83c91730735.svg?invert_in_darkmode" align=middle width=15.461490000000001pt height=31.056300000000004pt/>:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/4dfe14aee7ffc6427d772a515360eac2.svg?invert_in_darkmode" align=middle width=269.40869999999995pt height=38.72022pt/></p>

set
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/be175b1f9c187ee6451af89767590864.svg?invert_in_darkmode" align=middle width=130.845pt height=38.72022pt/></p>

we can rewrite <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/b189f0bc978e4a7853de920920cd11f4.svg?invert_in_darkmode" align=middle width=14.307150000000002pt height=22.745910000000016pt/> and <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/3166354ea06dfc8f7db7e83c91730735.svg?invert_in_darkmode" align=middle width=15.461490000000001pt height=31.056300000000004pt/>:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/5123df8f2b4e0da52635b12a2478b94c.svg?invert_in_darkmode" align=middle width=181.27724999999998pt height=110.41057499999998pt/></p>


### Several things to be noted:
- the posterior variance is the weighted average of prediction variance and measurement variance:

<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/77a9a84e9854a7d5f5b7c4e74160d53f.svg?invert_in_darkmode" align=middle width=454.67564999999996pt height=34.952279999999995pt/></p>

- the posterior variance is independent of either predicted value or observed value, it only depends on <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode" align=middle width=12.56145pt height=22.381919999999983pt/> and <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/993675a032f62874dedf17aa57ad2d57.svg?invert_in_darkmode" align=middle width=15.461490000000001pt height=26.90754000000001pt/>. So it can be computed before receiving the measurement.

### Parameter Learning in Kalman filter:
We can use the observed values from time 0 to time T to estimate the paramter of filter. This is a unsupervise learning problem. We can use EM algorithm to solve it:


#### E step:  
When we clamp the observation Z, and the paramters <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/3251ce71ffb279e5b56db2ae2ebfa541.svg?invert_in_darkmode" align=middle width=121.673475pt height=26.90754000000001pt/>, this filter becomes a distribution of hidden variable X from time 0 to T. We can map this distribution to a scalar by taking the expectation value:

<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/d40c84a30bc3c0cc910ddd1cdb9f84c1.svg?invert_in_darkmode" align=middle width=267.52275pt height=16.376943pt/></p>

We can use the forward-backward algorithm to calcualte the likelihood--the probability of states given observations and parameters. And because maximizing the expected log likelihood equals to maximizing the expected likelihood but easier to solve in the M step, so we look at the log version instead:

<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/0409f5cebb8451796ed1229f08bffbc3.svg?invert_in_darkmode" align=middle width=321.26325pt height=16.376943pt/></p>


Let 
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/07b0308759aeab73b5b58f84a11d1e1d.svg?invert_in_darkmode" align=middle width=448.17135pt height=16.376943pt/></p>

As we will see shortly in M step, maximizing the ELL needs the following quantities calculated for every time step:


<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/eed9a1af7a674e16dc822f89a6fbcf04.svg?invert_in_darkmode" align=middle width=173.68724999999998pt height=129.87117pt/></p>


**Forward message**:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/e61d8b8f5f536fb3bec8efd4adbe8628.svg?invert_in_darkmode" align=middle width=628.9057499999999pt height=126.27681pt/></p>

where <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/d8932de438d03e8967245034345adf4e.svg?invert_in_darkmode" align=middle width=58.16910000000001pt height=26.70657pt/> and <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/3d7b3789a4eb190d5bf64b74b5627528.svg?invert_in_darkmode" align=middle width=59.492895pt height=26.90754000000001pt/>


**Backward message**:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/fc259072c705a7cc0f0607fdd48a7cb0.svg?invert_in_darkmode" align=middle width=743.57085pt height=104.66180999999999pt/></p>


<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/897d1a2fae15b0b7a9abcfd5a4bd0aa3.svg?invert_in_darkmode" align=middle width=236.65125pt height=49.84914pt/></p>

where
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/47e3b0ae99d1781902214b4888d7f410.svg?invert_in_darkmode" align=middle width=250.95839999999998pt height=21.158444999999997pt/></p>



**So for <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/adad2506b8b45ba4d578873dda4d92e9.svg?invert_in_darkmode" align=middle width=22.535040000000002pt height=27.598230000000008pt/>**:  
We need to pass information forward using equation (1)(3) and (4) to calculate <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/fd444fb5623e35435efbe19b2d1cc5cf.svg?invert_in_darkmode" align=middle width=17.984175pt height=26.033369999999973pt/> for each time step and pass information backward using equation (7) to calcualte <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/adad2506b8b45ba4d578873dda4d92e9.svg?invert_in_darkmode" align=middle width=22.535040000000002pt height=27.598230000000008pt/> for each time step.
    
**So for <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/7d81403a617936bb633568bdbd04e47f.svg?invert_in_darkmode" align=middle width=20.487885000000002pt height=27.598230000000008pt/>**:  
Requires the information of <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/8625d1083c5d0fa3ea826960239d44a7.svg?invert_in_darkmode" align=middle width=22.704330000000002pt height=27.598230000000008pt/> and <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/adad2506b8b45ba4d578873dda4d92e9.svg?invert_in_darkmode" align=middle width=22.535040000000002pt height=27.598230000000008pt/>. Like calculating <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/adad2506b8b45ba4d578873dda4d92e9.svg?invert_in_darkmode" align=middle width=22.535040000000002pt height=27.598230000000008pt/>, we need to pass information forward using equation (3) and (5) to calculate <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/0f15e81af7c6ec94c42b776beaae2879.svg?invert_in_darkmode" align=middle width=18.153465000000004pt height=26.033369999999973pt/> for each time step and pass information backward using equation (8) to calcualte <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/8625d1083c5d0fa3ea826960239d44a7.svg?invert_in_darkmode" align=middle width=22.704330000000002pt height=27.598230000000008pt/> for each time step.
    
**So for <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/e799c613d8de897e9286b9cf1f3de81e.svg?invert_in_darkmode" align=middle width=40.59pt height=27.598230000000008pt/>**:  
<img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/e799c613d8de897e9286b9cf1f3de81e.svg?invert_in_darkmode" align=middle width=40.59pt height=27.598230000000008pt/> can be calculated passing information backward from time T using equation (11) and (9)


#### M step: 
Then we maximize the ELL treating the observation Z, and the hidden variable X as constant and paramters <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/efbeca25855631c1296bac3d6b82c56d.svg?invert_in_darkmode" align=middle width=185.805345pt height=26.90754000000001pt/> as variable. The optimal values for these parameters can be obtained by setting the derivative of ELL w.r.t. each of them to 0 and solve the equation.

We denote the paramters after maximization as <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/770acadb303d8f38b88bfbe8319a43b7.svg?invert_in_darkmode" align=middle width=100.87968pt height=21.802439999999976pt/>.

**Sensor model H**:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/21dec2730868fd84ed857c56f0f5652b.svg?invert_in_darkmode" align=middle width=331.5246pt height=105.04856999999998pt/></p>


**Sensor noise variance R**:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/d5edf751be159cf8324686afaa8f262d.svg?invert_in_darkmode" align=middle width=400.08374999999995pt height=105.04856999999998pt/></p>


**transition model F**:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/aeb6bac985485daeb77e3e0f132fa49e.svg?invert_in_darkmode" align=middle width=339.92805pt height=105.04856999999998pt/></p>


**Transition noise covariance Q**:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/c8ff9fddb48cfe43e90c6300431437fb.svg?invert_in_darkmode" align=middle width=465.7983pt height=162.51905999999997pt/></p>


**Initial state mean <img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/f30fdded685c83b0e7b446aa9c9aa120.svg?invert_in_darkmode" align=middle width=9.922935000000003pt height=14.102549999999994pt/>**:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/5b5bb2c00f073cb5887159a767b508a9.svg?invert_in_darkmode" align=middle width=202.4253pt height=61.524705pt/></p>


**Initial state covariance V**:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/922f1dcb75a4d314b0cd95a3e4227321.svg?invert_in_darkmode" align=middle width=381.68129999999996pt height=65.69607pt/></p>


#### Reference:
- Zoubin Ghahramani and Geoffrey, E. Hinton.(1996)  Paramter Estimation for Linear Dynamical Systems.

<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/e8099fe0a57304226c76b735cd236db2.svg?invert_in_darkmode" align=middle width=423.7398pt height=74.02725pt/></p>

And because maximize expected log likelihood equals to maxize expected likelihood but easier to solve,

<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/c1d7a99124ebb0b26c9dedf759fda20d.svg?invert_in_darkmode" align=middle width=612.3859499999999pt height=124.89576pt/></p>

As we have seen, the intial state variable, the probability representation of transition model and sensor model are:
<p align="center"><img src="https://rawgit.com/Xianlai/Xianlai/master/svgs/e757185f2e1e2f3598a0d66461611879.svg?invert_in_darkmode" align=middle width=438.74654999999996pt height=108.07236pt/></p>

























