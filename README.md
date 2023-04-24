# Data Science and eExplainability of Models Evaluation 

1. [Team Members and Responsibility](#team-members-and-responsibility)
2. [How to start the artifact](#how-to-start-the-artifact)
3. [What we started with](#what-we-started-with)
4. [How the code works](#how-the-code-works)
5. [Findings](#findings)
    1. [Initial Observations](#initial-observations)
    2. [Metrics](#metrics)
    3. [Results](#results)
    4. [Analysis](#analysis)
    5. [Hyperparameter Tuning](#hyperparameter-tuning)

(Link to video presentation)[https://youtu.be/TZhYO3PREvE]

## Team Members and Responsibility

Due to issues with another team, we had to form a group with only two members.

- 10583161
    - Data Cleaning
    - Logistic Regression
    - Nearest Neighbours
    - Decision Tree
    - XAI (SHAP) Implementation

- 10617467
    - Feature Engineering
    - Shallow Neural Network 
    - Naive Bayes
    - Support Vector Machine
    - Hyperparameter Tuning

## How to start the artifact

Simply load the Jupyter notebook file into an interpreter and run each cell - if Jupyter is not installed on your local machine, it can be opened in Google Collab and run there.

If running locally, you need to ensure all libraries are install. This can be done with the Python Package managers pip or conda.

## What we started with

We started with the Titanic dataset from Kaggle. The dataset contains 891 rows and 12 columns. The columns are as follows:

- PassengerId: Unique ID of the passenger
- Survived: Whether the passenger survived or not (0 = No, 1 = Yes)
- Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- Sex: Sex of the passenger (male or female)
- Age: Age of the passenger
- SibSp: Number of siblings/spouses aboard the Titanic
- Parch: Number of parents/children aboard the Titanic
- Ticket: Ticket number
- Fare: Passenger fare
- Cabin: Cabin number
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

Noteably, the 'Survived' column is the target variable and the remaining columns are the features. Additionally, Age in the data is fractional if less than 1. If the age is estimated, is it in the form of xx.5. Sibsp and Parch are the number of siblings/spouses and parents/children aboard the Titanic respectively.

We started looking at what models we could apply to the dataset to predict whether a passenger survived or not. [Ekinci et al. (2018)](https://www.researchgate.net/publication/324909545_A_Comparative_Study_on_Machine_Learning_Techniques_Using_Titanic_Dataset) compares the performance of 12 different classifiers. As we deemed the dataset relatively low dimensional and not large enough to apply deep learning models, we chose to compare the performance of the following models from the paper:

- Logistic Regression
- Decision Trees
- Nearest Neighbours
- Naive Bayes
- Support Vector Machines
- Neural Networks

[Ekinci et al. (2018)](https://www.researchgate.net/publication/324909545_A_Comparative_Study_on_Machine_Learning_Techniques_Using_Titanic_Dataset) also suggested feature engineering to improve the performance of the models.

## How the code works

The code looks at the effectiveness of a multitude of classifier models on the Titanic dataset.

Firstly, the code loads the 'train.csv' file from the Titanic dataset. Once loaded, the dataset is cleaned; removing null values and factoring fields like 'Sex' and 'Embarked' to integer values. 

Next, the code adds some features through feature engineering. The Sibsp and Parch columns are combined to create a new column called 'Family_Size' which is the total number of family members aboard the Titanic. The 'Cabin' column is also factored to create a new column called 'Deck' which is the deck of the cabin. The 'Fare' column is also factored to create a new column called 'Fare_Per_Person' which is the fare per person. The 'Age' column is also factored to create a new column called 'Age_Group' which is a boolean value indicating whether the passenger was a child or not. These new columns are then added to explore further relationships in the data. The structure of the resultant dataframe can be found below:

![Clean Dataframe](https://user-images.githubusercontent.com/39186016/233869367-03a52dec-c3c7-4ba2-a160-04f8150d0b23.png)

A series of plots are then created to explore the relationships between the features and the target variable 'Survived'. A heatmap is also created to explore the correlation between the features. To determine the significance of the correlations, a p-test is performed. PCA is attempted to reduce the dimensionality of the data to show a clear correlation or pattern but this is not successful.

Secondly, the code examines the performance of a multitude of classifier models. These models include: Logistic Regression, Decision Trees, Nearest Neighbours, Naive Bayes, Support Vector Machines and Neural Networks. The performance of each model is measured using the accuracy score, class likelihood ratios, ROC curves, DET curves and Precision-Recall curves. 

Next, the code compares the performance of the models using the above metrics. Finally, the code explores the optimisation of the two most promising models - Logistic Regression and Neural Networks. However, both models performed worse than the baseline model and so no further optimisation was performed.

## Findings

### Initial Observations

Looking at the dataset as a whole, the Principal Component Analysis (PCA) was not successful in reducing the dimensionality of the data. This is because the data is not linearly separable and so there is no clear correlation or pattern between the features. However, the Attributes Correlation Heatmap gives an insight into the features are most correlated to survival. It also highlighted the features that had been engineered as they had a high with original features, especially those directly derived from an original feature. After performing a Pearsone correlation test with a 0.05 significance level, we can see that all attributes apart from Family_size and SibSp are insignificant to an individuals survival. However, while the number of siblings/spouses seems not to be significant, the number of parents/children is. This implies that having a parent or child on board is more likely to increase your chances of survival than having a sibling or spouse. This is likely due to the fact that children were more likely to be prioritised for lifeboats than adults and had someone looking out for them.

![Principal Component Analysis](https://user-images.githubusercontent.com/39186016/233869244-818770d5-29d7-44c4-88a1-a69853867a70.png)

![Correlation Heatmap](https://user-images.githubusercontent.com/39186016/233869218-ceb0ca48-a139-411e-b052-b31cac6e66c2.png)

| **Feature** | _Class_ | _Sex_ | _Fare_ | _Age_  | _SibSp_ | _Parch_ | _Fare_ | _Embarked_ | _Deck_ | _Age_Group_ | _Family_Size_ | _Fare_Per_Person_ |
|-------------|---------|-------|--------|--------|---------|---------|--------|------------|--------|-------------|---------------|-------------------|
| **P Value** | 0.0     | 0.0   | 0.0    | 0.0278 | 0.6792  | 0.0011  | 0.0    | 0.0        | 0.0    | 0.0014      | 0.2297        | 0.0               |

Moreover, the phrase "Women and children first" was popularised by [Marshall (1912)](https://books.google.co.uk/books?id=xbxB0JI3OQ0C&q=women+and+children+first&redir_esc=y) and the data displayed in the following image corroborates that this was in fact followed on the Titanic; 75.3% of women on the Titanic survived in comparison to only 20.1% of men. Furthermore, 54% of children on the Titanic survived in comparison to only 37.9% of adults. The graph "Survival by Age" shows the distribution of people that survived by their age.

![Plots around Age and Sex](https://user-images.githubusercontent.com/39186016/233869253-2eefd617-b568-41ba-8c3a-b7fd990e23e5.png)

Furthermore, the data shows that the higher the deck, the higher the survival rate. One can assume this is because the higher the deck, the closer the passenger was to the lifeboats. The graph "Survival by Deck" shows the distribution of people that survived by their deck; every deck aside from Deck T had a survival rate greater than 50%. More interestingly, the graph "Deck by Class" shows the distribution of people by their deck and class. This shows that the higher the class, the higher the deck. This is because the higher the class, the more expensive the ticket and so the higher the deck. However, it does give us insight into the fact that the higher classes were given priority when boarding the lifeboats. This is backed up by the graph "Survival by Class".

![Plots around Class and Deck](https://user-images.githubusercontent.com/39186016/233869279-7cf4562a-84c8-418d-9441-cdf99dd46ae5.png)

The graph "Survival by Family Size" shows the distribution of people that survived by their family size, with a family size of 3 having the lowest mortality rate. This shows that the larger the family size, the lower the survival rate. This is because the larger the family size, the less likely it is that the family will be able to board a lifeboat together. Additionally, the graph "Family Size by Class" shows that the higher classes tended to travel with smaller family sizes. 

![Plots around Family Size](https://user-images.githubusercontent.com/39186016/233869302-72378016-8ba6-489a-a3ca-f1a75c3784ae.png)

Other plots were created, such as "Class by Embarked Location" and "Survival by Embarked Location" but these corroborate the finding above that the higher classes were given priority when boarding the lifeboats and thus have a higher survival rate.

![Plots around Embarked Location](https://user-images.githubusercontent.com/39186016/233869547-ce887f83-4d5f-4f58-bc0d-9be19c608781.png)

### Metrics

#### ROC Curve 

This shows the tradeoff between the true positive rate and false positive rate at different classification levels. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate. High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall).

#### DET Curve

This shows the tradeoff between the false alarm rate and miss rate. A high area under the curve represents both high recall and high precision, where high precision relates to a low false alarm rate, and high recall relates to a low miss rate. High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall).

#### Precision-Recall Curve

This shows the tradoff between precision and recall. The precision is the ratio of true positives to the sum of true and false positives. The recall is the ratio of true positives to the sum of true positives and false negatives. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate. High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall).

#### Accuracy 

This is the ratio of correctly predicted observation to the total observations. Accuracy is a good measure when the target classes are balanced, i.e. when the number of data points belonging to each class is about the same.

#### Class Liklihood Ratio

This is split into a Positive and a Negative Liklihood Ratio. The Positive Liklihood Ratio is the ratio of the probability of the positive class given the predictor, to the probability of the negative class given the predictor. The Negative Liklihood Ratio is the ratio of the probability of the negative class given the predictor, to the probability of the positive class given the predictor. A Positive Liklihood Ratio greater than 1 indicates that the predictor is more likely to predict a positive outcome, and a Negative Liklihood Ratio greater than 1 indicates that the predictor is more likely to predict a negative outcome.

### Results

![roc](https://user-images.githubusercontent.com/39186016/233873040-fa7afd12-0cb7-487d-af84-a8b667025f77.png)
![det](https://user-images.githubusercontent.com/39186016/233873041-bc08ef1a-51c1-4353-99bf-9bddc56d1ddf.png)
![pr](https://user-images.githubusercontent.com/39186016/233873039-4b2e0d56-c31b-4d36-9bb0-f01217d56f86.png)
![accuracy](https://user-images.githubusercontent.com/39186016/233874452-68cbd43d-c36a-4a9c-bf2c-66d080665ee5.png)
![likelihood](https://user-images.githubusercontent.com/45512716/233865155-e6331a4d-d02a-40c8-ae65-d85676b02b98.png)

### Analysis

##### Highest perfoming models

**Logistic Regression**

Both logistic regression and MLP models were found to achieve the highest accuracy among all the models tested. The visualizations  of the data indicated that certain features, such as gender and age, had significant discrepancies in their distribution between the survival and non-survival groups. This suggests that these features likely have significant coefficients, as a result, the logistic regression model had a easier time learning due to its ability to learn from significant feature coefficients.

**MLP**

The MLP model produced comparable results to the logistic regression model, we believe this is because of its ability to capture non-linear relationships between the input features and the outcome variable. The SHAP values for the MLP model reveal that the difference between the model's average predictions and expected value is small, indicating that the MLP model was able to identify which features were most important in making accurate predictions. Overall, these findings suggest that the MLP model was able to capture complex relationships between the input features and the outcome variable, leading to its strong performance.

![Log_mlp_shap](https://user-images.githubusercontent.com/45512716/233868895-814169b5-cabc-4cc0-9296-64dc29d6fdb9.png)

##### Lowest perfoming model

**K-Nearest Neighbours**

K-Nearest Neighbours was the model that performed the worst on the Titanic dataset. Since KNN is a distance-based algorithm, it may not be well-suited for datasets with heavy skewness, such as the Titanic dataset. In this case, a small subset of passengers with similar features were able to survive, while the majority of the passengers did not. Because everyone who survived was grouped closely together in feature space, the model was prone to overfitting and had difficulty making accurate predictions on new data. Additionally, KNN had the highest false positive rate among all the models tested, indicating that it had a tendency to incorrectly classify non-survivors as survivors.

![knn](https://user-images.githubusercontent.com/45512716/233869903-9d241b59-cad1-4fe1-8ff0-7a9069df71d2.png)


### Hyperparameter Tuning

The hyperparameters for the MLP and Logistic Regression models were tuned using a grid search. The grid search was performed using 5-fold cross-validation, and the best hyperparameters were selected based on the model's accuracy on the validation set. The best hyperparameters were found to be:

- MLP
    - activation: tanh
    - alpha: 0.0001
    - hidden_layer_sizes: (150, 100, 50)
    - learning_rate: adaptive
    - max_iter: 150
    - solver: adam

- Logistic Regression
    - C: 10
    - max_iter: 1000
    - penalty: l1
    - solver: liblinear

Despite the fact that these hyperparameters were found to be optimal for the Titanic dataset, they did not improve in performance. Both optimised models performed worse in all metrics than the default models as can be seen by the graphs below. However, they still performed better than all of the other types of default model, showing that for this dataset, Logistic Regression or MLP are the best optimisers.

![output](https://user-images.githubusercontent.com/39186016/233870599-a54f4a59-0c79-44fe-bf64-8f7dd4e2adf7.png)
![output1](https://user-images.githubusercontent.com/39186016/233870600-06216e4d-51b7-4581-9368-cca8e386e5d1.png)
![output2](https://user-images.githubusercontent.com/39186016/233870601-ab994166-dd07-4312-9c40-80817f60ed6b.png)
![output3](https://user-images.githubusercontent.com/39186016/233870602-b487b280-6eb7-495e-958f-1cf75776cfd4.png)
![output4](https://user-images.githubusercontent.com/39186016/233870603-4024d930-5c56-4816-bdd7-b862ffbd323c.png)
![output5](https://user-images.githubusercontent.com/39186016/233870604-165682e8-ca6a-47ee-b00c-e992948fe9b6.png)
![output6](https://user-images.githubusercontent.com/39186016/233870588-493da76c-77e9-4165-bb89-101dfccf551e.png)
![output7](https://user-images.githubusercontent.com/39186016/233870590-d10fb7f1-6002-4b16-83f1-6143a4bd6c43.png)
![output8](https://user-images.githubusercontent.com/39186016/233870593-dcb8dc98-c13c-4964-b3b7-479be433d2f5.png)
![output9](https://user-images.githubusercontent.com/39186016/233870595-61e09b8e-93d7-4ca9-af21-3ed96319232b.png)
![output10](https://user-images.githubusercontent.com/39186016/233870596-b3f176c9-58e7-4da7-9af6-53845e869320.png)
![output11](https://user-images.githubusercontent.com/39186016/233870597-5221b169-0dfa-423b-8424-a8271434a880.png)

This is because the hyperparameters were tuned using a grid search, which only explored the combination of hyperparameters we defined. As a result, the grid search may not be able to find the true optimal hyperparameters, only the optimal parameters out of the set given. In the future, we would like to explore more advanced hyperparameter tuning methods, such as Bayesian optimization, to find the optimal hyperparameters for the given dataset.
