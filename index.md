# Xian Lai

Architect -> Data scientist -> ...

## Contact:
Xian Lai   
Tel. : 917 593 3051   
E-mail : XianLaaai@gmail.com   

## Education:
Fordham University (Sept, 2016 - present)    
Master of Science in Data Analytics,   
Computer Information Science, GSAS.    
Graduate in May. 2018. GPA: 3.97/4.00      

Columbia University (2012 - 2013)    
Graduated School of Architecture.    
Master of Science in Advanced Architecture Design.   

## Knowledge & Skills:
- **Have the knowledge of discrete math, statistics, linear algebra and calculus.**     
Able to solve problems using mathematics knowledge including sets, propositional logic and first order logic, vectors and matrices manipulation and decomposition, various optimization methods, probablity and distributions, hypothesis testing.

- **Can preprocess datasets.**      

- **Have the knowledge and practical experience of unsupervised learning methods:**   
Able to implement or apply models or algorithms of linear matrix decompostion, non-linear manifold learning and graphical probability model inference to learn the informative or latent variables of dataset and reduce the dimensionality. Able to implement or apply clustering methods to learn internal structure of dataset.

- **Have the knowledge and practical experience of various supervised learning models:**   
Able to implement or apply learning algorithms like generallized linear models, SVM, tree models, neural networks etc on realword dataset and tune the hyperparameters. Able to evaluate the results or combine the decisions from various models appropriately.

- **Can solve problem using searching or planning methods, encode and inference using logic:**      
Able to implement or apply tree/graph search on problems with or without heuristic information or come up with an admissive heuristic. Able to use classic or graph planning to find optimal sequence of actions to achieve goal state. Able to encode and inference logical agent’s knowledge base.

- **Have the knowledge of database systems and distributed computing frameworks:**       
Understand how databases works and able to use SQL to store or query data. Understand and being able to develop algorithms making use of distributed computing frameworks like Apache Spark to analysis large datasets, streaming data in real-time.

- **Have the knowledge of data structures and algorithms.**    

- **Programming Language:**    
Python and data analysis related packages including pandas, numpy, scipy, sklearn and visualization packages like matplotlib, Bokeh etc;   
Matlab, R, SQL, bash.  

- **Other Applications:**        
QGIS, Adobe Creative Suite 6


## Experience & Related Works

**Personal Projects:**
- Learn features from tweets using Spark Streaming and visualize the results in real-time.    
    GitHub: https://github.com/Xianlai/streaming_tweet_feature_learning

- Clustering and visualize New York rental apartments with online rental posting data. 
    GitHub: https://github.com/Xianlai/Manhattan_rental_apartment_clustering 

- Classify online news’ popularity with various models and compare the performance.    
    GitHub: https://github.com/Xianlai/online_news_popularity_classification 

- Implement and visualize tree searching algorithms including BFS, DFS, A* search etc.   
    GitHub: https://github.com/Xianlai/Tree-Search-and-Visualization  

**Previous Working Experience:**
- Architectural Designer in RUR Architecture D.P.C, New York (2013 - 2016).

# Streaming Tweets Feature Learning with Spark

![](Xianlai/streaming_tweet_feature_learning/imgs/tweet_feature_learning.gif)

![](Xianlai/streaming_tweet_feature_learning/imgs/logos.png)
#### As the diagram showing above, this project implements a pipeline that learns predictive features from streaming tweets and visualizes the result in real-time.


- **Receive streaming tweets on master machine**:    
    Running the TweetsListener.py script in the background, a tweets stream with 3 tracks-"NBA", "NFL" and "MBL" are pulled from Tweepy API. Inside this stream, each tweet has a topic about one of those 3 tracks and is seperated by delimiter "><". 


- **Analysis tweets on distributed machines**:    
    This stream is directed into Spark Streaming API through TCP connection and distributed onto cluster. Under the Spark Streaming API, the distrbuted stream is abstracted as a data type called DStream. A series of operations are then applied on this DStream in real time and transform it into other DStreams containing intermediate or final analysis results. 
    
    1. preprocess each tweet into a label and a list of clean words which contains only numbers and alphabets.
    2. count the frequencies of words in all tweets and take the top 5000 most frequent ones as features.
    3. encode the tweets in last half minute into a structured dataset using features mentioned above.
    4. calculate the conditional probability given label and the predictiveness of each feature word.
    
    
- **Visualize results on master machine**:   
    At last we select the tweets and features with higher predictiveness, collect their label, sum of predictiveness and 2 tsne features back onto the master machine and visualize them as a scatter plot. This visualization can be used as a informal validation of predictiveness defined above. If the scatter of different are well seperated, then the features selected by this predictiveness measure are valid.
    
    1. keep only 300 most predictive features and discard other non-predictive features.
    2. calculate the sum of predictiveness of each word in tweet.
    3. take 60 tweets with the highest sum of predictiveness under each label.
    4. apply TSNE learning on these 300 data points to reduce dimentionality from 100 to 2 for visualization.
