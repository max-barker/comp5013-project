# comp5013-project
 
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

## How to start the artifact.

Simply load the Jupyter notebook file into an interpreter and run each cell - if Jupyter is not installed on your local machine, it can be opened in Google Collab and run there.

If running locally, you need to ensure all libraries are install. This can be done with the Python Package managers pip or conda.

## How the code works

The code looks at the Titanic dataset and attempts to predict whether a passenger survived or not. The code is split into 4 sections:

Firstly, the code loads the 'train.csv' file from the Titanic dataset. Once loaded, the dataset is cleaned; removing null values and factoring fields like 'Sex' and 'Embarked' to integer values. Next, engineered values are added such as 'Deck', 'Age_Group', 'Family_Size' and 'Fare_Per_Person' to explore further relationships in the data. A series of plots are then created to explore the relationships between the features and the target variable 'Survived'. A heatmap is also created to explore the correlation between the features. PCA is attempted to reduce the dimensionality of the data to show a clear correlation or pattern but this is not successful.

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

