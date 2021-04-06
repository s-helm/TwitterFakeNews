# TwitterFakeNews
This projekt aims to build classification models for an automatically collected Fake News dataset from Twitter.

## Data

The collected data the code refers to is available at http://data.dws.informatik.uni-mannheim.de/fakenews/ and includes the files listed below:
* *bow_models, d2v_models, tfidf_models, topic_models: bag-of-words, doc2vec, TF-IDF and topic* 
</br>models that were created from the training data and applied to the testing data
* *sample* contains the indices for the 20,000 or 100,000 samplesample that was used for some experiments: 
* *testdata*
  * *testset_tweet_user_features.csv* the manually collected testset with all features except Doc2Vec/BOW and topics
  * *testset_x.csv* full testset with all features including text and topic models where x is the name of the learner
* *text_data* contains the Doc2Vec and BOW component of the train- and testset
* *topic_data* contains the topic component of the train- and testset
* *data_set_tweet_user_features.csv* the 200,000 instances dataset with all tweet and user features. 

Initially the tweet data was collected into a MySQL Database that some code refers to. However for the machine learning experiments it is not required.

### Dataset Generation
The datasets for the different learners need to be generated from the three components (data_set_tweet_user_features + topics + text representation). Therefore the main method of the class DatasetUtils.DataSetCombiner can be executed. Combining also removes all attributes which are not used for learning. The dataset will then appear in the data folder.

## Learning
The class *Learning.LearningMain* is responsible for the evaluation of the learners. Depending on which line is commented in, one can also evaluate the testset. The most important parameters throughout the code are the following
* *all* refers to the use case (Use Case 2: all=0, Use Case 1: all=1)
* *clf_name* refers to the learner. (nb = Naïve Bayes, dt = Decision Tree, svm = SVM, nn = Neural Network, xgb = XGBoost, rf = Random Forest)


Voting and weighted voting can be performed in *Learning.Voting* for the whole dataset. The testset first has to be evaluated as described previously, then the respective line in the main method of *Learning.TestsetInterpretation* can be used. For weighted voting on the testset predict_proba needs to be true when evaluating in LearningMain.

Learning might consume a lot of memory. The results were created with 16GB of RAM. If sufficient memory and cores are available nr_of_cores can be increased in line 65 in Learning.XValidation

## Further packages
* *FeatureSelection* contains the methods for the feature selection on a variance threshold, feature importances and mutual information
* *ParameterTuning* includes the grid and random search for parameter tuning 
* *TextRepresentation.TextModelSelector* is responsible for creating and evaluating the text models with different configurations. 
* *TopicMining.TopicsSelector* is responsible for creating and evaluating the topic models with different configurations. 

Some feature generation steps were implemented in Java and are not included in this Repo.

## Libraries
The most important Python libraries used. Especially another Pandas version might cause errors.
- Pandas (0.19.2)
- Scikit learn (0.18.2)
- Numpy (1.12.0)
- Genism (2.1.0)

