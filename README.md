# Titanic-Survival-Prediction-Analyzing-Historical-Data-to-Predict-Passenger-Survival
Using ML to analyze passenger data from the Titanic disaster. we predicted survival of people based on various feature. we conducted data visualization, EDA, Feature Engineering and Label Encoding. Logistic Regression and Random Forest Classifier models were used then cross validation and hyperparameter tuning were used for optimization.

In this project, we are attempting to predict the survival outcomes of passengers aboard the RMS Titanic. The Titanic, a luxury passenger liner, famously sank after hitting an iceberg on its maiden voyage in 1912. The dataset used in this project contains information about various passengers, including their demographic details (such as age, gender, and passenger class) and whether they survived the disaster.

Our goal is to build a machine learning model that can accurately predict whether a given passenger survived or not based on the available data. By analyzing factors such as passenger demographics, family relationships, and cabin class, we aim to identify patterns and correlations that contribute to survival likelihood.

Summary:

In this project, we start by loading the training and test datasets and performing exploratory data analysis (EDA) to understand the characteristics of the data. Visualizations are employed to discern relationships between survival and different features such as gender, passenger class, and age group.

Feature engineering is then carried out to create new features from existing ones, such as categorizing ages into groups and extracting titles from passenger names. These engineered features provide additional insights into the data and enhance the predictive power of our models.

Next, we train multiple machine learning models, including Random Forest Classifier and Logistic Regression, to predict passenger survival. The models are evaluated using accuracy scores on a validation set, and the best performing model is selected for further optimization.

Hyperparameter tuning is performed on the chosen model to fine-tune its parameters and improve its predictive performance. This process involves systematically searching through different combinations of hyperparameters using techniques like grid search cross-validation.

Finally, the tuned model is applied to the test dataset to make predictions on unseen data. The predicted survival outcomes are then saved for analysis or submission.


Steps:

1. Data Loading and Preprocessing:
Loaded the training and test datasets from CSV files using pd.read_csv().
Performed basic exploration using train.head() to understand the structure of the data.
Data Visualization and Exploratory
2. Data Analysis (EDA):
Utilized sns.barplot() to visualize the relationship between survival and categorical variables like 'Sex' and 'Pclass'.
Identified patterns and insights from visualizations to understand the impact of different features on survival rates.
3. Feature Engineering:
Created a new feature 'AgeGroup' by categorizing ages into different groups using pd.cut() based on predefined bins and labels.
Extracted titles from the 'Name' column using regex and created a new feature 'Title'.
Simplified titles by grouping rare titles into broader categories using regex substitutions.
Encoded categorical features using LabelEncoder() from scikit-learn, transforming features like 'Sex', 'AgeGroup', and 'Embarked' into numerical values.
4. Model Training and Evaluation:
Split the dataset into features (X) and target (y).
Further split the data into training and validation sets using train_test_split() from scikit-learn.
Trained multiple models (Random Forest Classifier and Logistic Regression) using their respective scikit-learn implementations.
Evaluated model performance using accuracy score (accuracy_score() from scikit-learn) on the validation set.
Selected the best performing model based on accuracy.
5. Hyperparameter Tuning:
Defined hyperparameter grids for each model to search over during tuning.
Used GridSearchCV from scikit-learn to perform grid search cross-validation for hyperparameter tuning.
Tuned hyperparameters to optimize model performance using the best performing model selected earlier.
6. Final Model Validation:
Applied the best performing model (after hyperparameter tuning) on the test set to make predictions.
Generated predictions for the test set using the final model.
Optionally saved the predictions to a CSV file for further analysis or submission.


Through this project, we aim to gain insights into the factors influencing survival on the Titanic and develop a robust predictive model that can accurately classify passengers based on their likelihood of survival.
