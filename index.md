Last updated at March-08-2018

# Contact   
Email: XianLaaai@gmail.com   
Cell: 01-917-593-3051   
Addr: 311 W 55th St, NYC, NY, 10019   
[Resume](https://github.com/Xianlai/Xianlai.github.io/blob/master/resume.md)  

___
# Streaming Tweets Feature Learning with Spark
[Jump to repository](https://github.com/Xianlai/streaming_tweet_feature_learning)

![](imgs/logos.png)
#### As the diagram showing above, this project implements a pipeline that learns predictive features from streaming tweets and visualizes the result in real-time:

- **Receive streaming tweets on master machine**:    
    Running the TweetsListener.py script in the background, a tweets stream with 3 tracks-"NBA", "NFL" and "MBL" are pulled from Tweepy API. Inside this stream, each tweet has a topic about one of those 3 tracks and is seperated by delimiter "><". 

        example raw tweets:  
        Here's every Tom Brady Postseason TD! #tbt #NFLPlayoffs https://t.co/2CIHBpz2OW...  
        RT @ChargersRHenne: This guy seems like a class act.  I will root for him  
        RT @NBA: Kyrie ready! #Celtics #NBALondon https://t.co/KgZVsREGUK...  
        RT @NBA: The Second @NBAAllStar Voting Returns! https://t.co/urTwnGQNKl...  
        ...  

- **Analysis tweets on distributed machines**:    
    This stream is directed into Spark Streaming API through TCP connection and distributed onto cluster. Under the Spark Streaming API, the distrbuted stream is abstracted as a data type called DStream. A series of operations are then applied on this DStream in real time and transform it into other DStreams containing intermediate or final analysis results. 
    
    1. preprocess each tweet into a label and a list of clean words which contains only numbers and alphabets.
    
        ```
        example cleaned tweets after preprocessing:   
        tag:1, words:['rt', 'chargersrhenne', 'this', 'guy', ...],    
        tag:0, words:['rt', 'debruynekev', 'amp', 'ilkayguendogan', ...],    
        tag:0, words:['rt', 'commissioner', 'adam', 'silver', ...],    
        tag:0, words:['rt', 'spurs', 'all', 'star', ...],    
        tag:0, words:['nbaallstar', 'karlanthony', 'towns', 'nbavote', ...],   
        ...    
        ```

    2. count the frequencies of words in all tweets and take the top 5000 most frequent ones as features.
        
        ```
        example word count:  
        ('rt' , 196)  
        ('the', 174)  
        ('in' , 85)  
        ('for', 62)  
        ('to' , 59)  
        ...
        ```

    3. encode the tweets in last 15 seconds into a structured dataset using features mentioned above.
    
        ```
        example encoded dataset:  
        tag: 0, features: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],  
        tag: 1, features: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],  
        tag: 2, features: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],  
        tag: 0, features: [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, ...],  
        tag: 1, features: [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, ...],  
        ...  
        ```

    4. calculate the conditional probability given label and the predictiveness of each feature word.
        
        ```
        most predictive word : (cp0, cp1, cp2, pdtn)  
        allstar     : (0.14835164835164835, 0.0078125, 0.05555555555555555, 249.3439)  
        alabama     : (0.005494505494505495, 0.140625, 0.05555555555555555, 216.67129)  
        fitzpatrick : (0.005494505494505495, 0.1328125, 0.05555555555555555, 195.5925333333333)  
        voting      : (0.12637362637362637, 0.0078125, 0.05555555555555555, 187.45217333333335)  
        minkah      : (0.005494505494505495, 0.125, 0.05555555555555555, 175.55554999999998)  
        draft       : (0.016483516483516484, 0.171875, 0.1111111111111111, 149.7176)  
        ...  
        ```

- **Visualize results on master machine**:   
    At last we select the tweets and features with higher predictiveness, collect their label, sum of predictiveness and 2 tsne features back onto the master machine and visualize them as a scatter plot. 

    1. keep only 300 most predictive features and discard other non-predictive features.
    2. calculate the sum of predictiveness of each word in tweet.
    3. take 60 tweets with the highest sum of predictiveness under each label.
    4. apply TSNE learning on these 300 data points to reduce dimentionality from 100 to 2 for visualization.
    
    This visualization can be used as an informal way to validate the predictiveness defined above. If the scatter circles of different labels are well seperated, then the features selected by this predictiveness measure are working well.

![](imgs/tweet_feature_learning.gif)

## Files
- **[tweet_feature_learning_SparkStreaming.ipynb](https://github.com/Xianlai/streaming_tweet_feature_learning/blob/master/Spark_machine_learning_pipeline.ipynb)**  
    This jupyter notebook contains the code receiving tweets from socket, learn features and their stats and visualize selected tweets using learned features.

- **[TweetsListener.py](https://github.com/Xianlai/streaming_tweet_feature_learning/blob/master/TweetsListener.py)**  
    This python script pulls realtime tweets from tweepy API and forward it to the assigned TCP connect.(If you are not using docker container, you need to modify the IPaddress information in this file as well as in tweet_feature_learning_SparkStreaming.ipynb to make the streaming work.)

- **[StreamingPlot.py](https://github.com/Xianlai/streaming_tweet_feature_learning/blob/master/StreamingPlot.py)**  
    This python script implements the streaming plotting class which generate a scatter plotting and keeps updating the plotting with new plotting data source.

- **[logs.txt](https://github.com/Xianlai/streaming_tweet_feature_learning/blob/master/logs.txt)**   
    This text file is generated in tweet_feature_learning_SparkStreaming.ipynb to save intermediate and final analysis result.

- **[Spark_overview.md](https://github.com/Xianlai/streaming_tweet_feature_learning/blob/master/Spark_overview.md)**  
    This markdown file briefly introduces what is Spark and the functionalities of it.

- **[Spark_installation_guide.md](https://github.com/Xianlai/streaming_tweet_feature_learning/blob/master/pyspark_installation_guide.md)**  
    This markdown file contains guiding steps on how to install Spark, pyspark and how to run them either in the shell or as an application.

- **[Spark_AWS_cluster_guide.md](https://github.com/Xianlai/streaming_tweet_feature_learning/blob/master/Spark_AWS_cluster_guide.md)**  
    This markdown file contains guiding steps on how to set up a AWS cluster and how to run Spark on it.

- **[Spark_machine_learning_pipeline.ipynb](https://github.com/Xianlai/streaming_tweet_feature_learning/blob/master/Spark_machine_learning_pipeline.ipynb)**  
    This jupyter notebook implements an example using Spark machine learning API to classify whether online news are popular or not.


___
# Mahattan Rental Apartment Clustering
[Jump to repository](https://github.com/Xianlai/Manhattan_rental_apartment_clustering)

A city functions like a gigantic sophisticate network. Within it each buildings and blocks are connected by visible transportation systems and invisible functional dependencies. On the other hand, the difference of locations and functionality also divides the city into many sub-areas. 

For different purposes, the boundaries of these sub-areas are different. Like for political administration, we have boroughs, community districts and neighbourhoods, and for postal service, we have zip codes. 

In this projet, I made use of rental apartment online listing dataset and new york building footprint dataset to explore the possible geographic boundaries or patterns of apartment rental market. Equivalent to finding boundaries, clustering are performed to find the best grouping of buildings with respect to their location and rental market popularity and then we show how different properties like bedroom number, is there elevator in building, is there fitness center in building etc affect the clustering patterns.

![](imgs/prices.png)

This project is consist of 2 parts:

- **clustering model selection**  
    
    1. Interpolate the popularity of every building in the building dataset.

        Based on assumption that popularity of buildings are similar to their surrounding buildings', I use inverse distance weighting (IDW) as my interpolation method to get popularity value for each data point in building dataset from listing dataset.

    2. Cluster the buildings with location and popularity.
        
        With every building assigned popularity values, I performed hierarchical clustering using their longitude, latitude and the popularity.

    3. Evaluate the interpolation and clustering models with different parameter combination and select the best one for this project.
    
        In the previous 2 phases, there are 4 parameters: 

            - n_neighbors: number of neighbor building to consider during interpolating    
            - IDWpower: controlling power of IDW    
            - linkage: method to calculate distance between clusters    
            - metric: method to calculate distance between buildings    

        We use 6 criteria to evaluate each model. And through studying how these criteria behave and diverse, we choose whether to fuse score or rank of each scoring system for picking final model.

            - n_singlton : The number of singleton clusters.  
            - smClusterSize: The cluster size at the 15th percentile ranking from small to big.  
            - lgClusterSize: The cluster size at the 85th percentile ranking from small to big.  
            - lgClusterArea: The cluster area at the 85th percentile ranking from small to big.  
            - interVariance: The within cluster popularity variance.  
            - intraVariance: The between cluster popularity variance.  

- **Query clustering with different conditions**  
    
    1. Query clustering using cluster statistics

        In the process of clustering, we calculate some statistics for each cluster:

            - Popularity mean  
            - Popularity variance  
            - cluster size  
            - cluster area  

        We can either use them to filter clusters, (For example, we can filter out 100 clusters with highest popularities.) or use them as color coding to visualize these clusters. (For example, we can plot the clusters colored by their popularity mean.)
        
    2. Query clustering using different building properties
    
        Since the listing dataset contains information about building properties like price, fitness centers, bedroom numbers etc, we can produce different subset of listing data and interpolate the building popularity from this subset and hence get a different clustering. (For example, if we want to compare the clustering of high-price rentals to that of low-price rentals, we can create 2 subsets, get 2 clusterings and compare the difference in final plottings.)

## Files:
- **[1_model selection.ipynb](https://github.com/Xianlai/Manhattan_rental_apartment_clustering/blob/master/1_model%20selection.ipynb)**     
    shows the process of clustering model selection

- **[2_clustering.ipynb](https://github.com/Xianlai/Manhattan_rental_apartment_clustering/blob/master/2_clustering.ipynb)**    
    shows the how to make use of clustering to query information we are interested in and compare clusterings with different apartment properties.

- **[interactive_clusters.py](https://github.com/Xianlai/Manhattan_rental_apartment_clustering/blob/master/interactive_clusters.py)**    
    makes interactive plotting using bokeh server.

    Besides making static plotting, we can also query the clustering interactively with the help of bokeh server. Simply run `$ bokeh serve --show interactive_clusters.py` in the command line, the interactive plotting will be availabel at http://localhost:5006/, you can use web browser to play with it.

___
# Online News Popularity Classification
[Jump to repository](https://github.com/Xianlai/online_news_popularity_classification)

Facilitated by the fast spreading and developing of internet and smart devices, how can we understand online news browsing data and find the pattern inside it become more and more important.

In this project I am using online news popularity data set containing 39644 news articles and 57 features about each article including statistical features like number of words in title, rate of non-stop words in the content, article publish weekdays etc. and NLP features like positive word rate, title subjectivity level etc. The goal is to classify whether these articles are popular or not quantified by article shares. 

The main motivation of this project is setting up a systematic framework to:

    1. Understand the dataset including noises and possible hidden features can be extracted. 
    2. Visualize the behaviors of different learning models and observe how they interact with this dataset.
    3. Compare the behavior and performance of those learning models.

To limit the size of Jupyter notebooks, I split this project into 2 parts: preprocessing and model fitting selection.

- **Preprocessing**

    1. Explore the statistical figures like mean, std, range, unique value/outlier counts and feature data types of dataset.
    
    2. Clean the dataset by merging related binary features, standard scale features, remove outliers etc.   
        ![](imgs/feature_scales.png)

    3. Using matrix decomposition methods to reduce dimensionality and  generate possibly more predictive feature spaces.
    
        1. original
        2. PCA
        3. sparsePCA
        4. FactorAnalysis
        5. NMF
    
        ![](imgs/scatter_plot.png)


- **Model fitting and selection**

    1. Fit different learning models on the cleaned dataset under different feature space and test different hyper-parameter combinations using grid searching. 
    
        1. Naive Bayes
        2. Logistic Regression
        3. SVM
        4. k-Nearest Neighbours
        5. Random Forest
        6. XGBoost

    2. Visualize the results of parameter tunning to observe how each of the parameters changed the model's behavior on this dataset.
    
        ![](imgs/parallel_coordinates.png)

    3. Visualize the decision boundaries to tell how classifiers adapt themselves on this dataset and identify possible problems.
    
        ![](imgs/decision_boundaries.png)

    4. Evaluate and compare the performance of classifiers under different spaces using expected loss which considers both model flexibility and steadiness.
        
        ![](imgs/performance_compares.png)

    5. Compare the prediction "confidence" of classifiers by plotting the confusion histogram. This allows us further examine the behavior of classifiers and identify possible problems.
        
        ![](imgs/title_image_onp.png)


## Files:
- **[0_Preprocessing.ipynb](https://github.com/Xianlai/online_news_popularity_classification/blob/master/0_preprocessing.ipynb)**     
    This Jupyter notebook contains code preprocesses the original dataset.

- **[1_Model fitting and selection.ipynb](https://github.com/Xianlai/online_news_popularity_classification/blob/master/1_model_fitting_and_selection.ipynb)**     
    This Jupyter notebook contains code fits different learning models on cleaned dataset and compares the learning results.

- **[modules/LearningPipe.py](https://github.com/Xianlai/online_news_popularity_classification/blob/master/modules/LearningPipe.py)**  
    This python code implements a wrapper class based on several sklearn models to perform feature selection, grid searching, cross validation and evaluation jobs. It also provides methods to visualize parameter tuning and decision boundaries. 

- **[modules/Visual.py](https://github.com/Xianlai/online_news_popularity_classification/blob/master/modules/Visual.py)**  
    This python code implements visualization class for basic plotting jobs. Both python files are imported as modules in the Jupyter notebooks.


___
# General Tree Search and Visualization
[Jump to repository](https://github.com/Xianlai/Tree-Search-and-Visualization)

Tree search is one of the most flexible ways to deal with problems that can’t be solved directly and exactly. And an intuitive visualization of the paths found can certainly help us improve both the understanding of problem and solutions. This project implements a Python package that contains general tree search algorithms and result visualizaitons.

![](imgs/cover_tree_search.png)

## Modules
### A. Tree Search:
The tree search operations like expand node, evaluate state, append new nodes to tree as well as searching strategies like depth first search, breath first search etc. are implemented in the module TreeSearch which is used as the parent class of specific problem. It requires its children class instance to have the following methods:

```
- ProblemInstance._transition(state) 
    The transition model takes in state and return possible actions, result states and corresponding step costs.

- ProblemInstance._heuristicCost(state) 
    Calculate and return the heuristic cost given a state.

- ProblemInstance._isGoal(state) 
    Check whether given state is goal state or one of goal states.
```

### B.Tree Visualization:
In the visualization of result search tree, we use the polar coordinate system inspired by the following ideas:

1. As the search tree goes deeper, there are usually more nodes and thus requires more space. The polar coordinate system suits this need with increasing perimenter. So it’s more space efficient for showing the tree structure.

2. Each path from initial node to end node more or less forms a straight line. Thus it’s easier to identify the path and compare between different paths.


## How to use the modules
This package can be used through following API calls:
```
- ProblemInstance.breadthFirstSearch(maxNodes, maxLayers)  
- ProblemInstance.aStarSearch(maxNodes, maxLayers)  
- other search strategies...  

- ProblemInstance.plot_tree(diameter, background, title, ls, a)  
- ProblemInstance.print_paths()  
- ProblemInstance.export()  
```
See [TreeSearch_and_Visualization.ipynb](https://github.com/Xianlai/Tree-Search-and-Visualization/blob/master/TreeSearch_and_Visualization.ipynb) for usage examples.


## Files
- **[TreeSearch_and_Visualization.ipynb](https://github.com/Xianlai/Tree-Search-and-Visualization/blob/master/TreeSearch_and_Visualization.ipynb)**  
    This jupyter notebook contains the code illustrate how to use TreeSearch object and TreeVisual object to solve specific problem and visualize the result search tree.

- **[Documentations.md](https://github.com/Xianlai/Tree-Search-and-Visualization/blob/master/Documentations.md)**  
    This markdown file contains the documentation of TreeSearch, TreeVisual and RoadtripProblem classes.

- **[TreeSearch.py](https://github.com/Xianlai/Tree-Search-and-Visualization/blob/master/TreeSearch.py)**  
    This python script implements the general tree search algorithms. It includes the basic operations of tree search, like expand downward, trace backup etc, and different search strategies like BFS, DFS, A* etc. It should be used as parent class for specific problem instance.

- **[TreeVisual.py](https://github.com/Xianlai/Tree-Search-and-Visualization/blob/master/TreeVisual.py)**  
    This python script implements the class to visualize the result search tree. It includes the methods to parse the search tree in order to get plot data and the methods to plot the tree based on the attributes of its nodes like whether is goal node or whether is path. 

- **[RoadtripProblem.py](https://github.com/Xianlai/Tree-Search-and-Visualization/blob/master/RoadtripProblem.py)**  
    This python script implements an example problem of finding the best route in Romania to show the functions of its parent class--TreeSearch class.

![](imgs/breadth_first_search_light_background.png)


# Filters and Kalman Filter learning notes

### What are filters?
Filters are a kind of network models that incorporate the **certainty(our knowledge) and uncertainty(the noise in real world)** of our **belief and observation** for a dynamic system in a sequence of time steps. 

### What are filters used for?
For a dynamic system, if we are 100% confident about our knowledge, we can simply predict the state in any time step. Or if we are 100% confident about the observations, we can simply calculate the system state in any time step based on observations. 

But the real world is complex, we usually don't have full knowledge of the system and the observations usually contain certain amount of noise. So we need a way to incorporate our knowledge and observations in all time steps as much as possible. This is where we use filters.

### How do filters work?
The generic framework of a filter follows these steps:

1. **Guess** a initial system state. Because we are not 100% confident about our guess, we use a probability distribution to represent our belief;
2. **Receive** the observation at this time step --again this observation is uncertain, we use probability distribution to represent it-- and **combine** (*take a value between*) the information in our prediction and observation and update the system state at this time step(we are gaining information coming from observation);
3. **Guess** the state in next time step using our knowledge of this system. Because we are uncertain about our knowledge of this system, the uncertainty adds up. In other words, we are losing the confidence or information;
4. **Repeat** step 2-3 for following time steps.

**The essence of filter is the combination of prediction and measurement, which is a weighted average of these 2 values.** If we are more confident about our prediction, then the new value will be closer to our prediction value. If we are more confident about observation, then the new value bias toward observed value. 

![](imgs/network.png)


### Common variables names used in literature:

- ![x_t](https://latex.codecogs.com/gif.latex?x_t): actual state value at time t  
- ![\bar{x}_t](https://latex.codecogs.com/gif.latex?\bar{x}_t): state prior probability distribution at time t  
- ![\hat{x}_t](https://latex.codecogs.com/gif.latex?\hat{x}_t): state posterior probability distribution at time t   


- ![z_t](https://latex.codecogs.com/gif.latex?z_t): actual observed value at time t  
- ![\bar{z}_t](https://latex.codecogs.com/gif.latex?\bar{z}_t): prior probability distribution of observed variable predicted from ![\bar{x}_t](https://latex.codecogs.com/gif.latex?\bar{x}_t)  
- ![\hat{z}_t](https://latex.codecogs.com/gif.latex?\hat{z}_t): posterior probability distribution of observed variable given ![z_t](https://latex.codecogs.com/gif.latex?z_t)


- ![P_t](https://latex.codecogs.com/gif.latex?P_t): state variance, which is increasing in prediction step and decreaasing in update step. 
    + ![\bar{P}_t](https://latex.codecogs.com/gif.latex?\bar{P}_t): the prior state variance
    + ![\hat{P}_t](https://latex.codecogs.com/gif.latex?\hat{P}_t): the posterior state variance
- ![Q](https://latex.codecogs.com/gif.latex?Q): process noise, part of transition model, which typically won't change.  
- ![R](https://latex.codecogs.com/gif.latex?R): measure noise, part of sensor model, which typically won't change.  


- ![F](https://latex.codecogs.com/gif.latex?F): transition model
- ![H](https://latex.codecogs.com/gif.latex?H): sensor model


### From a probabilistic point of view:

1. Guess the prior probability distribution of system state at <img src="https://latex.codecogs.com/gif.latex?$t_0$" title="$t_0$" />:

    <img src="https://latex.codecogs.com/gif.latex?P(\bar{x}_0)" title="P(\bar{x}_0)" />

2. Receive observation <img src="https://latex.codecogs.com/gif.latex?P(z_0)" title="P(z_0)" /> at <img src="https://latex.codecogs.com/gif.latex?$t_0$" title="$t_0$" /> and combine this observation as a posterior probability distribution with our guess using Bayesian theorem:  

    <img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;P(\hat{x}_0)&space;&&space;=&space;P(x_0|z_0)\\&space;&&space;=&space;\frac{P(z_0|\bar{x}_0)P(\bar{x}_0)}{P(z_0)}\\&space;&&space;=&space;\frac{P(z_0|\bar{x}_0)P(\bar{x}_0)}&space;{\sum_{\bar{x}_0}&space;P(z_0,&space;\bar{x}_0)}&space;\end{aligned}" title="\begin{aligned} P(\hat{x}_0) & = P(x_0|z_0)\\ & = \frac{P(z_0|\bar{x}_0)P(\bar{x}_0)}{P(z_0)}\\ & = \frac{P(z_0|\bar{x}_0)P(\bar{x}_0)} {\sum_{\bar{x}_0} P(z_0, \bar{x}_0)} \end{aligned}" />

3. Guess the prior probability distribution of system state at <img src="https://latex.codecogs.com/gif.latex?$t_1$" title="$t_1$" />: 

    <img src="https://latex.codecogs.com/gif.latex?P(\bar{x}_1)&space;=&space;\sum_{\hat{x}_0}(P(\bar{x}_1|\hat{x}_0)P(\hat{x}_0))" title="P(\bar{x}_1) = \sum_{\hat{x}_0}(P(\bar{x}_1|\hat{x}_0)P(\hat{x}_0))" />
    
4. Repeat step 2-3 for following time steps.

Note that the conditional probability <img src="https://latex.codecogs.com/gif.latex?P(z_0|\bar{x}_0)" title="P(z_0|\bar{x}_0)" /> contains the knowledge of how system state generate observations(sensor model). It includes both situations when the observations are directly measurement of system state and when they are not(they are actually measurements of a related but different state).

And the conditional probability <img src="https://latex.codecogs.com/gif.latex?P(\bar{x}_t|\hat{x}_{t-1})" title="P(\bar{x}_t|\hat{x}_{t-1})" /> contains the knowledge of how system state evolve to next state(transition model).


### *Addition and Multiplication of probability distributions:

- **Addition**:[Wikipedia](https://en.wikipedia.org/wiki/Convolution_of_probability_distributions)

    It means when 2 values adding up(<img src="https://latex.codecogs.com/gif.latex?$Z&space;=&space;X&space;&plus;&space;Y$" title="$Z = X + Y$" />), if X has probability distribution <img src="https://latex.codecogs.com/gif.latex?$P(X)$" title="$P(X)$" /> and Y has probability distribution <img src="https://latex.codecogs.com/gif.latex?$P(Y)$" title="$P(Y)$" />, then Z has probability distribution <img src="https://latex.codecogs.com/gif.latex?$P(Z)$" title="$P(Z)$" /> which is the convolution of <img src="https://latex.codecogs.com/gif.latex?$P(X)$" title="$P(X)$" /> and <img src="https://latex.codecogs.com/gif.latex?$P(Y)$" title="$P(Y)$" />.

    <img src="https://latex.codecogs.com/gif.latex?P(Z=z)&space;=&space;\int_{-\infty}^{\infty}&space;P(X&space;=&space;x)P(Y&space;=&space;z-x)&space;dx" title="P(Z=z) = \int_{-\infty}^{\infty} P(X = x)P(Y = z-x) dx" />
    
- **Multiplicaiton**: [Wikipedia](https://en.wikipedia.org/wiki/Product_distribution)   

    Similar, if X and Y are two independent, continuous random variables, described by probability density functions $P(X)$ and $P(Y)$ then the probability density function of $Z = XY$ is:

    <img src="https://latex.codecogs.com/gif.latex?P(Z=z)&space;=&space;\int_{-\infty}^{\infty}&space;P(X&space;=&space;x)&space;P(Y&space;=&space;\frac{z}{x})&space;\frac{1}{\lvert&space;x&space;\rvert}&space;dx" title="P(Z=z) = \int_{-\infty}^{\infty} P(X = x) P(Y = \frac{z}{x}) \frac{1}{\lvert x \rvert} dx" />

#### Addtion and Multiplication of Gaussian distributions
Fortunately, the addition and multiplicaiton of Gaussian distributions are quite easy:

- **Addition**:

    <img src="https://latex.codecogs.com/gif.latex?\mathcal{N}(\mu_1,&space;\sigma_1^2)&space;&plus;&space;\mathcal{N}(\mu_2,&space;\sigma_2^2)&space;=&space;\mathcal{N}(\mu_1&space;&plus;&space;\mu_2,&space;\sigma_1^2&space;&plus;&space;\sigma_2^2)" title="\mathcal{N}(\mu_1, \sigma_1^2) + \mathcal{N}(\mu_2, \sigma_2^2) = \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)" />

    
    
- **Linear transformation**:    
    We can treat this as addtion multiple times.

    <img src="https://latex.codecogs.com/gif.latex?a\mathcal{N}(\mu,&space;\sigma^2)&space;&plus;&space;b&space;=&space;\mathcal{N}(a\mu&space;&plus;&space;b,&space;a^2\sigma^2)" title="a\mathcal{N}(\mu, \sigma^2) + b = \mathcal{N}(a\mu + b, a^2\sigma^2)" />

- **Multiplication**:
    
    <img src="https://latex.codecogs.com/gif.latex?\mathcal{N}(\mu_1,&space;\sigma_1^2)&space;*&space;\mathcal{N}(\mu_2,&space;\sigma_2^2)&space;=&space;\mathcal{N}(&space;\frac{\sigma_2^2\mu_1&space;&plus;&space;\sigma_1^2\mu_2}{\sigma_1^2&space;&plus;&space;\sigma_2^2},&space;\frac{\sigma_1^2&space;\sigma_2^2}{\sigma_1^2&space;&plus;&space;\sigma_2^2})" title="\mathcal{N}(\mu_1, \sigma_1^2) * \mathcal{N}(\mu_2, \sigma_2^2) = \mathcal{N}( \frac{\sigma_2^2\mu_1 + \sigma_1^2\mu_2}{\sigma_1^2 + \sigma_2^2}, \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2 + \sigma_2^2})" />


## What is Kalman filter?

Kalman filters are a special kind of filters which parameterize the previous probability distribution as Gaussian distributions: we assume next state is a linear transformation of previous state add Gaussian noise: 

<img src="https://latex.codecogs.com/gif.latex?x_{t&plus;1}&space;=&space;Fx_t&space;&plus;&space;w,&space;w&space;\sim&space;\mathcal{N}(0,&space;Q)" title="x_{t+1} = Fx_t + w, w \sim \mathcal{N}(0, Q)" />

equivalently 
<img src="https://latex.codecogs.com/gif.latex?P(x_{t&plus;1}|x_t)&space;=&space;\mathcal{N}(Fx_t,&space;Q)" title="P(x_{t+1}|x_t) = \mathcal{N}(Fx_t, Q)" />


### In cases where observations are directly measures of system state:
We have observed variable:

<img src="https://latex.codecogs.com/gif.latex?$$z_t&space;=&space;x_t&space;&plus;&space;v,&space;v&space;\sim&space;\mathcal{N}(0,&space;R)$$" title="$$z_t = x_t + v, v \sim \mathcal{N}(0, R)$$" />

equivalently 

<img src="https://latex.codecogs.com/gif.latex?$$P(z_t|x_t)&space;=&space;\mathcal{N}(x_t,&space;R)$$" title="$$P(z_t|x_t) = \mathcal{N}(x_t, R)$$" />

**Prior State**:

<img src="https://latex.codecogs.com/gif.latex?$$&space;P(\bar{x}_t)&space;\sim&space;\mathcal{N}(\bar{x}_t,&space;\bar{P}_t)&space;$$" title="$$ P(\bar{x}_t) \sim \mathcal{N}(\bar{x}_t, \bar{P}_t) $$" /> 

**FORWARD Update**:

<img src="https://latex.codecogs.com/gif.latex?$$\begin{aligned}&space;P(\hat{x}_t)&space;\sim&space;\mathcal{N}(\hat{x}_t,&space;\hat{P}_t)&space;&&space;=&space;P(z|\bar{x}_t,&space;R)&space;\mathcal{N}(\bar{x}_t,&space;\bar{P}_t)&space;&&&space;(1)&space;\\&space;&&space;=&space;P(\bar{x}_t|z,&space;R)&space;\mathcal{N}(\bar{x}_t,&space;\bar{P}_t)&space;&&&space;(2)&space;\\&space;&&space;=&space;\mathcal{N}(z_t,&space;R)&space;\mathcal{N}(\bar{x}_t,&space;\bar{P}_t)&space;\\&space;&&space;=&space;\mathcal{N}(&space;\frac{\bar{P}_t&space;z_t&space;&plus;&space;R&space;\bar{x}_t}{\bar{P}_t&space;&plus;&space;R},&space;\frac{\bar{P}_t&space;R}{\bar{P}_t&space;&plus;&space;R})&space;\\&space;&&space;=&space;\mathcal{N}(&space;\frac{\bar{P}_t}{\bar{P}_t&space;&plus;&space;R}&space;z_t&space;&plus;&space;\frac{R}{\bar{P}_t&space;&plus;&space;R}\bar{x}_t,&space;\frac{\bar{P}_t&space;R}{\bar{P}_t&space;&plus;&space;R})&space;\\&space;&&space;=&space;\mathcal{N}(K&space;z_t&space;&plus;&space;(I-K)\bar{x}_t,&space;KR)&space;\\&space;&&space;=&space;\mathcal{N}(\bar{x}_t&space;&plus;&space;K(z_t&space;-&space;\bar{x}_t),&space;KR)&space;\\&space;\end{aligned}$$" title="$$\begin{aligned} P(\hat{x}_t) \sim \mathcal{N}(\hat{x}_t, \hat{P}_t) & = P(z|\bar{x}_t, R) \mathcal{N}(\bar{x}_t, \bar{P}_t) && (1) \\ & = P(\bar{x}_t|z, R) \mathcal{N}(\bar{x}_t, \bar{P}_t) && (2) \\ & = \mathcal{N}(z_t, R) \mathcal{N}(\bar{x}_t, \bar{P}_t) \\ & = \mathcal{N}( \frac{\bar{P}_t z_t + R \bar{x}_t}{\bar{P}_t + R}, \frac{\bar{P}_t R}{\bar{P}_t + R}) \\ & = \mathcal{N}( \frac{\bar{P}_t}{\bar{P}_t + R} z_t + \frac{R}{\bar{P}_t + R}\bar{x}_t, \frac{\bar{P}_t R}{\bar{P}_t + R}) \\ & = \mathcal{N}(K z_t + (I-K)\bar{x}_t, KR) \\ & = \mathcal{N}(\bar{x}_t + K(z_t - \bar{x}_t), KR) \\ \end{aligned}$$" />

where

<img src="https://latex.codecogs.com/gif.latex?K&space;=&space;\frac{\bar{P}_t}{\bar{P}_t&space;&plus;&space;R}" title="K = \frac{\bar{P}_t}{\bar{P}_t + R}" />

\* (1): here the likelihood <img src="https://latex.codecogs.com/gif.latex?P(z|\bar{x}_t,&space;R)" title="P(z|\bar{x}_t, R)" /> use sensor noise  
\* (2): <img src="https://latex.codecogs.com/gif.latex?P(z|\bar{x}_t,&space;R)&space;=&space;P(\bar{x}_t|z,&space;R)" title="P(z|\bar{x}_t, R) = P(\bar{x}_t|z, R)" /> because <img src="https://latex.codecogs.com/gif.latex?d(\bar{x}_t,z_t)&space;=&space;d(z_t,&space;\bar{x}_t)" title="d(\bar{x}_t,z_t) = d(z_t, \bar{x}_t)" />


**FORWARD Predict**:

<img src="https://latex.codecogs.com/gif.latex?$$\begin{aligned}&space;P(\bar{x}_{t&plus;1})&space;\sim&space;\mathcal{N}(\bar{x}_{t&plus;1},&space;\bar{P}_{t&plus;1})&space;&&space;=&space;\int_{\hat{x}_t}&space;P(\bar{x}_{t&plus;1}|\hat{x}_t)&space;P(\hat{x}_t)&space;d\hat{x}_t\\&space;&&space;=&space;\int_{\hat{x}_t}&space;P(\bar{x}_{t&plus;1}|F\hat{x}_t)&space;P(F\hat{x}_t)&space;d\hat{x}_t&space;&&&space;(1)\\&space;&&space;=&space;\int_{\hat{x}_t}&space;\mathcal{N}(\bar{x}_{t&plus;1};&space;F&space;x'_t,&space;Q)&space;\mathcal{N}(F&space;x'_t;&space;F\hat{x}_t,&space;F\hat{P}_t&space;F^\intercal&space;)&space;d\hat{x}_t&space;&&&space;(2)\\&space;&&space;=&space;\int_{\hat{x}_t}&space;\mathcal{N}(\bar{x}_{t&plus;1}&space;-&space;F&space;x'_t;&space;0,&space;Q)&space;\mathcal{N}(F&space;x'_t;&space;F\hat{x}_t,&space;F&space;\hat{P}_t&space;F^\intercal)&space;d\hat{x}_t&space;&&&space;(3)\\&space;&&space;=&space;\mathcal{N}(\bar{x}_{t&plus;1};&space;F\hat{x}_t,&space;F&space;\hat{P}_t&space;F^\intercal&space;&plus;&space;Q)&&&space;(4)\\&space;\end{aligned}$$" title="$$\begin{aligned} P(\bar{x}_{t+1}) \sim \mathcal{N}(\bar{x}_{t+1}, \bar{P}_{t+1}) & = \int_{\hat{x}_t} P(\bar{x}_{t+1}|\hat{x}_t) P(\hat{x}_t) d\hat{x}_t\\ & = \int_{\hat{x}_t} P(\bar{x}_{t+1}|F\hat{x}_t) P(F\hat{x}_t) d\hat{x}_t && (1)\\ & = \int_{\hat{x}_t} \mathcal{N}(\bar{x}_{t+1}; F x'_t, Q) \mathcal{N}(F x'_t; F\hat{x}_t, F\hat{P}_t F^\intercal ) d\hat{x}_t && (2)\\ & = \int_{\hat{x}_t} \mathcal{N}(\bar{x}_{t+1} - F x'_t; 0, Q) \mathcal{N}(F x'_t; F\hat{x}_t, F \hat{P}_t F^\intercal) d\hat{x}_t && (3)\\ & = \mathcal{N}(\bar{x}_{t+1}; F\hat{x}_t, F \hat{P}_t F^\intercal + Q)&& (4)\\ \end{aligned}$$" />

\* (1): after linear transformation  
\* (2): where <img src="https://latex.codecogs.com/gif.latex?x_t'" title="x_t'" /> is a value at time t  
\* (3): move first Gaussian to original to match the convolution equation  
\* (4): and the convolution of 2 Gaussians is the addtion of them   

### In cases where observations are NOT directly measured on system state:

We assume observation <img src="https://latex.codecogs.com/gif.latex?z_t" title="z_t" /> is a linear transformation of state <img src="https://latex.codecogs.com/gif.latex?x_t&space;:&space;z_t&space;=&space;Hx_t" title="x_t : z_t = Hx_t" />. So we need to adjust the forward update function:

**The prior distribution of observed variable** <img src="https://latex.codecogs.com/gif.latex?\bar{z}_t" title="\bar{z}_t" />:

<img src="https://latex.codecogs.com/gif.latex?P(\bar{z}_t)&space;=&space;\mathcal{N}(H\bar{x}_t,&space;H\bar{P}_tH^\intercal)" title="P(\bar{z}_t) = \mathcal{N}(H\bar{x}_t, H\bar{P}_tH^\intercal)" />


And the **likelihood**:

<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;P(z_t|\bar{z}_t)&space;&&space;=&space;\mathcal{N}(z_t;&space;\bar{z}_t,&space;R)\\&space;&&space;=&space;\mathcal{N}(z_t;&space;H\bar{x}_t,&space;R)\\&space;&&space;=&space;\mathcal{N}(H\bar{x}_t;&space;z_t,&space;R)\\&space;&&space;=&space;\mathcal{N}(z_t,&space;R)&space;\end{aligned}" title="\begin{aligned} P(z_t|\bar{z}_t) & = \mathcal{N}(z_t; \bar{z}_t, R)\\ & = \mathcal{N}(z_t; H\bar{x}_t, R)\\ & = \mathcal{N}(H\bar{x}_t; z_t, R)\\ & = \mathcal{N}(z_t, R) \end{aligned}" />


Thus we can calculate the **posterior observed variable distribution**:

<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;P(\hat{z}_t)&space;&&space;=&space;P(\bar{z}_t)&space;P(z_t|\bar{z}_t)\\&space;&&space;=&space;\mathcal{N}(H\bar{x}_t,&space;H&space;\bar{P}_t&space;H^\intercal)&space;\mathcal{N}(z_t,&space;R)\\&space;&&space;=&space;\mathcal{N}(&space;\frac{R&space;H\bar{x}_t&space;&plus;&space;H&space;\bar{P}_t&space;H^\intercal&space;z_t}{H&space;\bar{P}_t&space;H^\intercal&space;&plus;&space;R},&space;\frac{H&space;\bar{P}_t&space;H^\intercal&space;R}{H&space;\bar{P}_t&space;H^\intercal&space;&plus;&space;R}&space;)&space;\end{aligned}" title="\begin{aligned} P(\hat{z}_t) & = P(\bar{z}_t) P(z_t|\bar{z}_t)\\ & = \mathcal{N}(H\bar{x}_t, H \bar{P}_t H^\intercal) \mathcal{N}(z_t, R)\\ & = \mathcal{N}( \frac{R H\bar{x}_t + H \bar{P}_t H^\intercal z_t}{H \bar{P}_t H^\intercal + R}, \frac{H \bar{P}_t H^\intercal R}{H \bar{P}_t H^\intercal + R} ) \end{aligned}" />


Because <img src="https://latex.codecogs.com/gif.latex?$P(\hat{z}_t)&space;\sim&space;\mathcal{N}(H\hat{x}_t,&space;H\hat{P}_tH^\intercal)$" title="$P(\hat{z}_t) \sim \mathcal{N}(H\hat{x}_t, H\hat{P}_tH^\intercal)$" />,

<img src="https://latex.codecogs.com/gif.latex?$$&space;H\hat{x}_t&space;=&space;\frac{R&space;H&space;\bar{x}_t&space;&plus;&space;H&space;\bar{P}_t&space;H^\intercal&space;z_t}{H&space;\bar{P}_t&space;H^\intercal&space;&plus;&space;R}&space;$$" title="$$ H\hat{x}_t = \frac{R H \bar{x}_t + H \bar{P}_t H^\intercal z_t}{H \bar{P}_t H^\intercal + R} $$" />

and

<img src="https://latex.codecogs.com/gif.latex?$$&space;H&space;\hat{P}_t&space;H^\intercal&space;=&space;\frac{H&space;\bar{P}_t&space;H^\intercal&space;R}{H&space;\bar{P}_t&space;H^\intercal&space;&plus;&space;R}&space;$$" title="$$ H \hat{P}_t H^\intercal = \frac{H \bar{P}_t H^\intercal R}{H \bar{P}_t H^\intercal + R} $$" />

Solve for <img src="https://latex.codecogs.com/gif.latex?$\hat{x}_t$" title="$\hat{x}_t$" /> and <img src="https://latex.codecogs.com/gif.latex?$\hat{P}_t$" title="$\hat{P}_t$" />:

<img src="https://latex.codecogs.com/gif.latex?$$\begin{aligned}&space;\hat{x}_t&space;&&space;=&space;\frac{R&space;\bar{x}_t&space;&plus;&space;\bar{P}_tH^\intercal&space;z_t}{H\bar{P}_tH^\intercal&space;&plus;&space;R}\\&space;\hat{P}_t&space;&&space;=&space;\frac{\bar{P}_t&space;R}{H\bar{P}_tH^\intercal&space;&plus;&space;R}&space;\end{aligned}$$" title="$$\begin{aligned} \hat{x}_t & = \frac{R \bar{x}_t + \bar{P}_tH^\intercal z_t}{H\bar{P}_tH^\intercal + R}\\ \hat{P}_t & = \frac{\bar{P}_t R}{H\bar{P}_tH^\intercal + R} \end{aligned}$$" />

set

<img src="https://latex.codecogs.com/gif.latex?$$&space;K_t&space;=&space;\frac{\bar{P}_tH^\intercal}{H\bar{P}_tH^\intercal&space;&plus;&space;R}&space;$$" title="$$ K_t = \frac{\bar{P}_tH^\intercal}{H\bar{P}_tH^\intercal + R} $$" />

we can rewrite <img src="https://latex.codecogs.com/gif.latex?\hat{x}_t" title="\hat{x}_t" /> and <img src="https://latex.codecogs.com/gif.latex?\hat{P}_t" title="\hat{P}_t" />:

<img src="https://latex.codecogs.com/gif.latex?$$\begin{aligned}&space;\hat{x}_t&space;&&space;=&space;K_t&space;z_t&space;&plus;&space;(1&space;-&space;HK_t)\bar{x}_t&space;\\&space;&&space;=&space;\bar{x}_t&space;&plus;&space;K_t&space;(z_t&space;-&space;H\bar{x}_t)&space;\\&space;\hat{P}_t&space;&&space;=&space;\bar{P}_t&space;\frac{R}{H\bar{P}_tH^\intercal&space;&plus;&space;R}&space;\\&space;&&space;=&space;(I&space;-&space;K_tH)&space;\bar{P}_t&space;\end{aligned}$$" title="$$\begin{aligned} \hat{x}_t & = K_t z_t + (1 - HK_t)\bar{x}_t \\ & = \bar{x}_t + K_t (z_t - H\bar{x}_t) \\ \hat{P}_t & = \bar{P}_t \frac{R}{H\bar{P}_tH^\intercal + R} \\ & = (I - K_tH) \bar{P}_t \end{aligned}$$" />


### Several things to be noted:
- the posterior variance is the weighted average of prediction variance and measurement variance:

<img src="https://latex.codecogs.com/gif.latex?$$&space;\frac{HPH^\intercal}{HPH^\intercal&space;&plus;&space;R}&space;\text{(real&space;observation)}&space;&plus;&space;\frac{R}{HPH^\intercal&space;&plus;&space;R}&space;\text{(prior&space;observation)}&space;$$" title="$$ \frac{HPH^\intercal}{HPH^\intercal + R} \text{(real observation)} + \frac{R}{HPH^\intercal + R} \text{(prior observation)} $$" />

- the posterior variance is independent of either predicted value or observed value, it only depends on <img src="https://latex.codecogs.com/gif.latex?$R$" title="$R$" /> and <img src="https://latex.codecogs.com/gif.latex?$\bar{P}_t$" title="$\bar{P}_t$" />. So it can be computed before receiving the measurement.

## Reference:
- Roger R. Labbe, Kalman and Bayesian Filters in Python [Kalman and Bayesian Filters in Python](http://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb)
- Zoubin Ghahramani and Geoffrey, E. Hinton.(1996)  Paramter Estimation for Linear Dynamical Systems.










