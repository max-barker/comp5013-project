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

Next, the code adds some features through feature engineering. The Sibsp and Parch columns are combined to create a new column called 'Family_Size' which is the total number of family members aboard the Titanic. The 'Cabin' column is also factored to create a new column called 'Deck' which is the deck of the cabin. The 'Fare' column is also factored to create a new column called 'Fare_Per_Person' which is the fare per person. The 'Age' column is also factored to create a new column called 'Age_Group' which is a boolean value indicating whether the passenger was a child or not. These new columns are then added to explore further relationships in the data. 

A series of plots are then created to explore the relationships between the features and the target variable 'Survived'. A heatmap is also created to explore the correlation between the features. PCA is attempted to reduce the dimensionality of the data to show a clear correlation or pattern but this is not successful.

Secondly, the code examines the performance of a multitude of classifier models. These models include: Logistic Regression, Decision Trees, Nearest Neighbours, Naive Bayes, Support Vector Machines and Neural Networks. The performance of each model is measured using the accuracy score, class likelihood ratios, ROC curves, DET curves and Precision-Recall curves. 

Next, the code compares the performance of the models using the above metrics. Finally, the code explores the optimisation of the two most promising models - Logistic Regression and Neural Networks. However, both models performed worse than the baseline model and so no further optimisation was performed.

## Findings

### Metrics
- Accuracy
- Recall
- ROC curve


### Results
![results1](https://user-images.githubusercontent.com/45512716/233865183-236dcad5-8b4a-4c38-8920-658990b6020f.png)
![results 2](https://user-images.githubusercontent.com/45512716/233865155-e6331a4d-d02a-40c8-ae65-d85676b02b98.png)

### Analysis

Both logistic regression and MLP models were found to achieve the highest accuracy among all the models tested. The visualizations of the data indicated that certain features, such as gender and age, had significant discrepancies in their distribution between the survival and non-survival groups. This suggests that these features likely have significant coefficients, as a result, the logistic regression model had a easier time learning due to its ability to learn from significant feature coefficients.

the MLP model produced comparable results to the logistic regression model, we believe this is because of its ability to capture non-linear relationships between the input features and the outcome variable. The SHAP values for the MLP model reveal that the difference between the model's average predictions and expected value is small, indicating that the MLP model was able to identify which features were most important in making accurate predictions. Overall, these findings suggest that the MLP model was able to capture complex relationships between the input features and the outcome variable, leading to its strong performance.
![shap](https://user-images.githubusercontent.com/45512716/233868423-f35ae2e8-1ce2-42e1-8bf2-56b5182b8776.png)


