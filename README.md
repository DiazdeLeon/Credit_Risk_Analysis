# 1.Overview of the loan prediction risk analysis

The purpose of this analysis is to apply machine learning to solve credit card risk. Also, to employ different techniques to train and evaluate models with unbalanced classes. It will be use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. Then it will be oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Next it will use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Finally, it will compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. The last step is to evaluate the performance of these models and with a written recommendation on whether they should be used to predict credit risk.

# 2. Results

The "loan status" was used to determine whether the application was considered "low" or "high" risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk". The dataset is about 68,840 total applications with 99% classified as "low risk".
Using the 75/25 method to split the data for training vs. testing, 68,470 "low risk" and 347 "high risk" applications were categorized into the training set.

## 2.1	RandomOverSampler
Resampling involves creating a new transformed version of the training dataset in which the selected examples have a different class distribution. This is a simple and effective strategy for imbalanced classification problems. Applying re-sampling strategies to obtain a more balanced data distribution is an effective solution to the imbalance problem. This model RandomOverSampler, involves randomly selecting examples from the minority class, with replacement, and adding them to the training dataset.

### 2.1.1	Accuracy score 
The balance accuracy score is 65%

![image](https://user-images.githubusercontent.com/95872614/167276312-5e6375cf-21c6-4843-84f3-0af79940fb50.png)

### 2.1.2 Confusion matrix and imbalanced classification report 

![image](https://user-images.githubusercontent.com/95872614/167276125-9b7c9f6a-96de-4b63-b70a-58f244b368e6.png)

- The precision of the model is 54/(54+5435) = 0.009 this value is quite low, it is not possible to assure whether it is a  sure high risk or low.
- The Sensitivity of the model is 54/(54+33) = 0.62 this value is also quite low, it may a have a consequence of leaving out many possible high-risk credits.
- The F1 score is 2*(0.62*0.009)/(0.62+0.009) = 0.017, it si possible to conclude that there is bias between precision and sensitivity.
- In one hand, "High Risk" precision rate was only 1% with the recall at 62% giving this model an F1 score of 2%. In the other hand, "Low Risk" had a precision rate of 100% and recall at 68%.

## 2.2 The Synthetic Minority Oversampling Technique Model also known as SMOTE. 

Increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection. 

![image](https://user-images.githubusercontent.com/95872614/167276135-97633701-6246-4bba-a0c8-ca54e6ef81c7.png)

### 2.2.1 Accuracy score 

The balance accuracy score is 62%

![image](https://user-images.githubusercontent.com/95872614/167276329-cbf4f0b9-a5a0-412e-8d93-14145e60afdc.png)

### 2.2.2 Confusion matrix and imbalanced classification report

![image](https://user-images.githubusercontent.com/95872614/167276154-5d3dd1cd-6b6e-454f-bfa6-5ad1bc58d2db.png)

- The precision of the model is 51/(51+5783) = 0.009 this value is significantly low, it is not possible to assure whether it is a  sure high risk or low.
- The Sensitivity of the model is 51/(51+36) = 0.59 this value is also quite low, it may a have a consequence of leaving out many possible high-risk credits.
- The F1 score is 2*(0.59*0.009)/(0.59+0.009) = 0.017, it is possible to conclude that there is bias between precision and sensitivity.
- It is no possible to choose between RandomOverSampler and SMOTE due to both models have weaknesses. 

## 2.3	ClusterCentroids

This model makes undersampling by generating a new set based on centroids by clustering methods. The algorithm is generating a new set according to the cluster centroid of a KMeans algorithm.

![image](https://user-images.githubusercontent.com/95872614/167276176-4cb60e57-c2d9-4534-8751-d63ea2844b5f.png)

### 2.3.1 Accuracy score 

![image](https://user-images.githubusercontent.com/95872614/167276183-f5b1dc16-163e-4a79-9966-829fe5c0ca35.png)

The balance accuracy score is 52%

#### 2.3.2 Confusion matrix and imbalanced classification report

![image](https://user-images.githubusercontent.com/95872614/167276194-ee61b9d8-6b1c-4a33-95cb-18ed8fbab3e9.png)

- The precision of the model is 52/(52+9684) = 0.005 this value is significantly low, it is not possible to assure whether it is a  sure high risk or low.
- The Sensitivity of the model is 52/(52+35) = 0.60 this value is also quite low, it may a have a consequence of leaving out many possible high-risk credits.
- The F1 score is 2*(0.60*0.005)/(0.60+0.005) = 0.099, it is possible to conclude that there is bias between precision and sensitivity.
- It is no possible to choose between ClusterCentroids and the two previous models (RandomOverSampler and SMOTE), as matter of fact the figures look even not as good as expected.

## 2.4	SMOTEENN

This model combines the SMOTE ability to generate synthetic examples for minority class and ENN ability to delete some observations from both classes that are identified as having different class between the observation’s class and its K-nearest neighbor majority class.

![image](https://user-images.githubusercontent.com/95872614/167276210-7632b1bd-e7e1-465a-96b6-8f4a855a58ca.png)

### 2.4.1 Accuracy score 

![image](https://user-images.githubusercontent.com/95872614/167276217-c97fdb65-c876-48c6-9d15-cedac25143fd.png)

The balance accuracy score is 64%

### 2.4.2 Confusion matrix and imbalanced classification report

![image](https://user-images.githubusercontent.com/95872614/167276224-acf1249b-237b-495f-b1c9-933005b7328a.png)

- The precision of the model is 61/(61+7262) = 0.008 this value is significantly low, it is not possible to assure whether it is a  sure high risk or low.
- The Sensitivity of the model is 61/(61+26) = 0.70 this value is also quite low, it may a have a consequence of leaving out many possible high-risk credits.
- The F1 score is 2*(0.70*0.008)/(0.70+0.008) = 0.015, it is possible to conclude that there is bias between precision and sensitivity.

## 2.5 BalancedRandomForestClassifier

Compaing the two new Machine Learning models that reduce bias to predict credit risk. The models classified 51,366 as High Risk and 246 as Low Risk.

BalancedRandomForestClassifier Model, two trees of the same size and equal size to the minority class are constructed to represent one for the majority class and one for the minority class.

### 2.5.1. Accuracy score

The balanced accuracy score increased to 78.9% for this model.

![image](https://user-images.githubusercontent.com/95872614/167276246-8eb247e7-a15a-44e7-833d-d970c4195715.png)

### 2.5.2 Confusion matrix and imbalanced classification report

![image](https://user-images.githubusercontent.com/95872614/167276250-3b706e46-39d0-43b2-8c61-2ff0229f1486.png)


The precision of the model is 71/(71+2153) = 0.003 this value is significantly low, it is not possible to assure whether it is a  sure high risk or low.
- The Sensitivity of the model is 71/(71+30) = 0.70 this value is also quite low, it may a have a consequence of leaving out many possible high-risk credits.
- The F1 score is 2*(0.70*0.003)/(0.70+0.003) = 0.006, it is possible to conclude that there is bias between precision and sensitivity.

## 2.6 EasyEnsembleClassifier

A set of classifiers where individual decisions are combined to classify new examples.

### 2.6.1 Accuracy score

The balanced accuracy score increased to 93.2% for this model.

![image](https://user-images.githubusercontent.com/95872614/167276267-24e7c886-3579-43c7-abdd-b8d3ae9ef2b6.png)

2.6.2 Confusion matrix and imbalanced classification report

![image](https://user-images.githubusercontent.com/95872614/167276273-c9cf3bda-82c0-4a3b-ae61-0453741face5.png)

- The precision of the model is 93/(93+983) = 0.086 this value is low, it is not possible to assure whether it is a  sure high risk or low.
- The Sensitivity of the model is 93/(93+8) = 0.92 this value is also quite better, it may a have a consequence of leaving out many possible high-risk credits.
- The F1 score is 2*(0.92*0.086)/(0.92+0.086) = 0.157, it is possible to conclude that there is bias between precision and sensitivity.

# 3.	Summary:

After the revision of the six models, it is concluded that the EasyEnsembleClassifer model is the best model due to, it produces the best results: accuracy rate of 93.2% and a 9% precision rate when predicting "High Risk candidates. The sensitivity rate was also the highest at 92% compared to the other models. The result for predicting "Low Risk" was also the highest with the sensitivity rate at 94% and an F1 score of 97%. Therefore, if a model needed to be recommended to perform this type of analysis, then this one would be the clear choice.










