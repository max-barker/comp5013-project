# Data Science and eExplainability of Models Evaluation 
 
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

Looking at the dataset as a whole, the Principal Component Analysis (PCA) was not successful in reducing the dimensionality of the data. This is because the data is not linearly separable and so there is no clear correlation or pattern between the features. However, the Attributes Correlation Heatmap gives an insight into the features are most correlated to survival. It also highlighted the features that had been engineered as they had a high with original features, especially those directly derived from an original feature. After performing a Pearsone correlation test with a 0.05 significance level, we can see that all attributes apart from Family_size and SibSp are significant to an individuals survival.  

![Principal Component Analysis](https://user-images.githubusercontent.com/39186016/233869244-818770d5-29d7-44c4-88a1-a69853867a70.png)

![Correlation Heatmap](https://user-images.githubusercontent.com/39186016/233869218-ceb0ca48-a139-411e-b052-b31cac6e66c2.png)

Moreover, the phrase "Women and children first" was popularised by [Marshall (1912)](https://books.google.co.uk/books?id=xbxB0JI3OQ0C&q=women+and+children+first&redir_esc=y) and the data displayed in the following image corroborates that this was in fact followed on the Titanic; 75.3% of women on the Titanic survived in comparison to only 20.1% of men. Furthermore, 54% of children on the Titanic survived in comparison to only 37.9% of adults. The graph "Survival by Age" shows the distribution of people that survived by their age.

![Plots around Age and Sex](https://user-images.githubusercontent.com/39186016/233869253-2eefd617-b568-41ba-8c3a-b7fd990e23e5.png)

Furthermore, the data shows that the higher the deck, the higher the survival rate. One can assume this is because the higher the deck, the closer the passenger was to the lifeboats. The graph "Survival by Deck" shows the distribution of people that survived by their deck; every deck aside from Deck T had a survival rate greater than 50%. More interestingly, the graph "Deck by Class" shows the distribution of people by their deck and class. This shows that the higher the class, the higher the deck. This is because the higher the class, the more expensive the ticket and so the higher the deck. However, it does give us insight into the fact that the higher classes were given priority when boarding the lifeboats. This is backed up by the graph "Survival by Class".

![Plots around Class and Deck](https://user-images.githubusercontent.com/39186016/233869279-7cf4562a-84c8-418d-9441-cdf99dd46ae5.png)

The graph "Survival by Family Size" shows the distribution of people that survived by their family size, with a family size of 3 having the lowest mortality rate. This shows that the larger the family size, the lower the survival rate. This is because the larger the family size, the less likely it is that the family will be able to board a lifeboat together. Additionally, the graph "Family Size by Class" shows that the higher classes tended to travel with smaller family sizes. 

![Plots around Family Size](https://user-images.githubusercontent.com/39186016/233869302-72378016-8ba6-489a-a3ca-f1a75c3784ae.png)

Other plots were created, such as "Class by Embarked Location" and "Survival by Embarked Location" but these corroborate the finding above that the higher classes were given priority when boarding the lifeboats and thus have a higher survival rate.

![Plots around Embarked Location](https://user-images.githubusercontent.com/39186016/233869547-ce887f83-4d5f-4f58-bc0d-9be19c608781.png)

### Metrics
- Accuracy
- Recall
- ROC curve


### Results
![results1](https://user-images.githubusercontent.com/45512716/233865183-236dcad5-8b4a-4c38-8920-658990b6020f.png)
![results 2](https://user-images.githubusercontent.com/45512716/233865155-e6331a4d-d02a-40c8-ae65-d85676b02b98.png)

### Analysis
- Logistic Regression

Both logistic regression and MLP models were found to achieve the highest accuracy among all the models tested. The visualizations of the data indicated that certain features, such as gender and age, had significant discrepancies in their distribution between the survival and non-survival groups. This suggests that these features likely have significant coefficients, as a result, the logistic regression model had a easier time learning due to its ability to learn from significant feature coefficients.

- MLP

the MLP model produced comparable results to the logistic regression model, we believe this is because of its ability to capture non-linear relationships between the input features and the outcome variable. The SHAP values for the MLP model reveal that the difference between the model's average predictions and expected value is small, indicating that the MLP model was able to identify which features were most important in making accurate predictions. Overall, these findings suggest that the MLP model was able to capture complex relationships between the input features and the outcome variable, leading to its strong performance.

![Log_mlp_shap](https://user-images.githubusercontent.com/45512716/233868895-814169b5-cabc-4cc0-9296-64dc29d6fdb9.png)

- K-Nearest Neighbours
K-Nearest Neighbours was the model that performed the worst on the Titanic dataset. Since KNN is a distance-based algorithm, it may not be well-suited for datasets with heavy skewness, such as the Titanic dataset. In this case, a small subset of passengers with similar features were able to survive, while the majority of the passengers did not. Because everyone who survived was grouped closely together in feature space, the model was prone to overfitting and had difficulty making accurate predictions on new data. Additionally, KNN had the highest false positive rate among all the models tested, indicating that it had a tendency to incorrectly classify non-survivors as survivors.

