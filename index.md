# Streaming Tweets Feature Learning with Spark

![](https://github.com/Xianlai/streaming_tweet_feature_learning/tree/master/imgs/tweet_feature_learning.gif)

![](https://github.com/Xianlai/streaming_tweet_feature_learning/tree/master/imgs/logos.png)
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
