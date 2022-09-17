# Name: Prashaanth Muthuraman
# Student Number: R00194750

# Storing the file path of the Stroke prediction dataset in separate 
# variables for better readability.

filePath1 = 'C:/Users/prash/Downloads/MSc Data Science and Analytics (Sem 2)/'

filePath2 = 'Applied Machine Learning/Project 2'

# Saving the overall project directory in a string variable.

projectDir = filePath1 + filePath2

# Importing the 'os' package for file handling operations.

import os

# Changing the working directory where the Stroke prediction dataset is saved.

os.chdir(projectDir)

# Importing the 'pandas' package for working with dataframes.

import pandas as pd

# Storing the Stroke prediction data in a Pandas dataframe variable.

strokeDataFrameOriginal = pd.read_csv('healthcare-dataset-stroke-data.csv')

# The 'id' column can be removed from the dataset.

strokeDataFrame = strokeDataFrameOriginal.copy()

strokeDataFrame = strokeDataFrame.drop(columns = ['id'])

# Displaying the structural information of the Stroke prediction Dataset.

strokeDataFrame.info()

# Before we perform Exploratory Data Analysis (EDA) we need to identify the 
# data that has categorical information. In the dataset, the fields 
# "hypertension", "heart_disease" and "stroke" are actually binary categorical 
# data despite holding numeric values. These fields must be converted to 
# string type.

# Storing the predictor names of the dataset in a temporary list.

tempNamesStroke = ["hypertension", "heart_disease", "stroke"]

# Converting the numeric fields to string.

for tempNameStroke in tempNamesStroke:
    
    # Storing the converted data in a variable. If there is any missing data,
    # it would also be converted as string. For example, nan or na would 
    # become "nan" or "na" respectively.
    
    strokeNum2Str = strokeDataFrame[tempNameStroke].astype(str)
    
    # Replacing the numeric data with string data in the dataset.
    
    strokeDataFrame[tempNameStroke] = strokeNum2Str
    
# Storing the column names of Stroke prediction dataset in a variable.
 
columnNamesStroke = list(strokeDataFrame.columns.values)

# The "pandas.api.types" package contains functions that can check the 
# datatype of a variable in a pandas dataframe or Series.

import pandas.api.types as ty

# Since there would be some visualisation involved, we would be using the 
# "matplotlib.pyplot" and "seaborn" packages.

import matplotlib.pyplot as plt

import seaborn as sns

# 1. Exploratory data analysis (EDA)

# EDA is done to understand the relationship between the variables.
# This gives us insights on the data which can be used for building predictive
# models. 

# Plotting Histograms, Boxplots and Barplots for understanding the distribution
# of data. 

for columnNameStroke in columnNamesStroke:
    
    # Storing the column value of the dataset in a variable.
    
    colValStroke = strokeDataFrame[columnNameStroke]
    
    # Checking if the column contains integer or float values.
    
    boolInt = ty.is_integer_dtype(colValStroke)
    
    boolFloat = ty.is_float_dtype(colValStroke)
    
    if boolInt or boolFloat:
        
        # Generating the histogram for numeric columns. 
        
        plt.figure()
        
        plot1 = sns.distplot(colValStroke)
        
        # Adding title to the plot.
        
        plt.title('Histogram of variable "' + columnNameStroke + '"')
        
        # Adding the xlabel to the plot.
        
        plot1.set_xlabel(columnNameStroke)
        
        # Adding the ylabel to the plot.
        
        plot1.set_ylabel('Frequency')
        
        # Displaying the plot.
        
        plt.show()
        
        # Generating the BoxPlot for numeric columns. 
        
        plt.figure()
        
        # Storing the Box plot title in a string variable.
        
        boxPlotTitle = 'Boxplot of variable "' + columnNameStroke + '"'
        
        colValStroke.plot.box(title = boxPlotTitle)
        
        # Displaying the plot.
        
        plt.show()
        
    # Checking if the column contains other data types such as string.
        
    else:
        
        plt.figure()
        
        # Storing the bar plot title in a string variable.
        
        barPlotTitle = 'Barplot of variable "' + columnNameStroke + '"'
        
        # Generating the barplot for categorical columns.
        
        colValStroke.value_counts().plot(kind = "bar"
                                         , title = barPlotTitle
                                         , rot = 0)
                
        # Adding the xlabel to the plot.
        
        plt.xlabel(columnNameStroke)
        
        # Adding the ylabel to the plot.
        
        plt.ylabel('Frequency')
        
        # Displaying the plot.
            
        plt.show()

# 1.1. Univariate Analysis.

# Here, univariate plots are created to understand the behaviour of each field.
        
# From the barplot of the "gender" variable, it is observed that majority of 
# the patients are female while the rest are male with one patient's gender 
# as 'other'.

# The histogram of the "age" variable is quite resemblant to a normal 
# distribution with a minor negative skew. The boxplot shows that the age has
# zero outliers.

# From the barplots of the "hypertension" and "heart_disease" variables, it is 
# observed that most of the patients do not suffer from both hypertension as 
# well as any heart disease. A small percentage of people suffer from both.

# From the barplot of the "ever_married" variable, it is observed that most of 
# the patients are married except for a small fraction of people.

# From the barplot of the "work_type" variable, it is observed that a high 
# percentage of the people work in private organizations. Around 16% of the 
# population are self-employed workers. Roughly 13% of the population 
# consist of children and people working for the goverment. Only 4% of the
# patients have never worked.

# From the barplot of the "Residence_type" variable, it is observed that 
# the percentage of people residing in rural areas is almost equal to the 
# percentage of people residing in urban localities.

# The histograms of the "avg_glucose_level" and "bmi" variables are quite 
# resemblant to a normal distribution with a strong positive skew. 
# The boxplots show that both the glucose level and bmi have multiple outliers 
# which needs to be handled in the preprocessing stage.

# From the barplot of the "smoking_status" variable, it is observed that a high 
# percentage (37%) of the people have never smoked. Around 17% of the patients 
# used to smoke before quitting. Roughly 16% of the people still smoke while
# it is still unknown whether the remainder of the people smoke or not.

# From the barplot of the "stroke" variable, it is observed that a high 
# percentage (95%) of the people have not experienced any stroke. Only 5% of 
# the patients have suffered from stroke.

# Creating a list for storing the numeric and categorical fields separately.

numColList = []

catColList = []

for columnNameStroke in columnNamesStroke:
    
    # Storing the column value of the dataset in a variable.
    
    colValStroke = strokeDataFrame[columnNameStroke]
    
    # Checking if the column contains integer or float values.
    
    boolInt = ty.is_integer_dtype(colValStroke)
    
    boolFloat = ty.is_float_dtype(colValStroke)
    
    if boolInt or boolFloat:
        
        numColList.append(columnNameStroke)
                
    # Checking if the column contains other data types such as string.
        
    else:
        
        catColList.append(columnNameStroke)

# 1.2. Bivariate Analysis.
        
# A Correlation Matrix can be used to understand the bivariate relationship 
# among the numerical fields.

# A temporary dataframe is created which holds the numeric data only.

tempStrokeDf = strokeDataFrame.copy()

tempStrokeDf = tempStrokeDf.drop(columns = catColList)

# Creation of the correlation matrix.

corrMat = tempStrokeDf.corr()

# Plotting the Correlation Matrix using a heat map.

sns.heatmap(corrMat,cmap = sns.diverging_palette(250, 15, as_cmap = True))

# From the Correlation Matrix, we can see that there is a weak relationship
# between the variables 'age' and 'avg_glucose_level' and a slightly
# weaker relationship between the pairs 'age' and 'bmi'. Compared to these
# relationships, the relationship between variables 'avg_glucose_level' and 
# 'bmi' is slightly stronger.

# 2. Data Pre-Processing.

# Before model building, pre-processing is the most important step.
# This involves dealing with outliers, dealing with missing values, 
# handling categorical data, scaling data, handling imbalance, feature selection
# and dimensionality reduction.

# 2.1. Dealing with missing values.

# Checking the fields containing missing values.

print(strokeDataFrame.isnull().sum())

# It is observed that only the 'bmi' field contains missing values.

# Storing the count of missing data in a variable.

missingCnt = strokeDataFrame.isnull().sum()['bmi']/(strokeDataFrame.shape)[0]

print('The percentage of missing data is %0.2f' % (missingCnt*100))

# Since the percentage of missing data is quite less (4%), there would not be
# much impact if those rows are removed.

strokeDataFrameCleaned = strokeDataFrame.dropna()

# 2.2. Dealing with outliers.

# Outliers are data that differ significantly from other data in a sample. 
# Outliers skew the data distributions and impacts the basic statistical 
# measures and can be responsible for underperformance of certain algorithms.
# Some ML algorithms are relatively robust to outliers. 
# Other ML algorithms (such as multiple linear regression) are much more 
# sensitive to outliers.

# The boxplots of the "avg_glucose_level" and "bmi" variables show that both 
# the glucose level and bmi have multiple outliers which needs to be handled in 
# the preprocessing stage.

# Since there are two variables possessing outliers, it is better to use a
# common clustering technique called DBScan. DBScan relies on the idea that 
# clusters are dense, so it starts exploring the data space in all directions 
# and marks a cluster boundary when the density of points decreases.
# Areas of data space with insufficient density of points are just considered 
# to be outliers or noise. 

# DBScan is a clustering technique that expects the data to be standardized.

# Importing the packages required for performing DBSCAN.

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

for j in range(10,120,10):
    
    strokeDataFrameCleaned = strokeDataFrame.dropna()
    
    # Standardizing the feature data.
    
    scaler = StandardScaler().fit(strokeDataFrameCleaned[numColList])
    
    strokeDfTransform = scaler.transform(strokeDataFrameCleaned[numColList])
    
    # Removal of outliers using DBSCAN.
    
    DB = DBSCAN(eps = 0.5,min_samples = j)
    
    DB.fit(strokeDfTransform)
    
    for i in range(len(numColList)):

        strokeDataFrameCleaned[numColList[i]] = strokeDfTransform[:,i]
    
    strokeDataFrameCleaned = strokeDataFrameCleaned[DB.labels_ !=-1]
    
    for numCol in numColList:
        
        # Generating the BoxPlot for numeric columns. 
        
        plt.figure()
        
        # Storing the Box plot title in a string variable.
        
        strTitle = 'Boxplot of variable "' + numCol + '" after outlier removal'
        
        strTitle2 = ' with min_samples = ' + str(j)
        
        strokeDataFrameCleaned[numCol].plot.box(title = strTitle + strTitle2)
        
        # Displaying the plot.
        
        plt.show()
    
# The 'min_samples' refers to the number of samples (or total weight) in a 
# neighborhood for a point to be considered as a core point. 
# This includes the point itself. From the boxplots we can see that as 
# 'min_samples' increases, the samples get closer which removes the outliers.
# The outliers are removed when min_samples = 110.

# 2.3. Handling Categorical Data.

# For building models, it is essential to convert categorical data into
# numeric data. Categorical data can be either ordinal or nominal.
# Ordinal data have logical ordering and there is no automatic mechanism for
# encoding this. Nominal data have no inherent ordering and can be encoded
# easily using algorithms such as 'one-hot encoding'.
# This technique will create new dummy features for each unique value in the 
# nominal feature column.

# Storing the predictor columns in a list.

predColumns = columnNamesStroke[:len(columnNamesStroke)-1]

# Storing the response column name and data in separate variables.

respCol = columnNamesStroke[len(columnNamesStroke)-1]
 
# Encoding the predictore data using 'one-hot encoding'.

strokeDataFrameEncoded = pd.get_dummies(strokeDataFrameCleaned[predColumns])

strokeDataFrameEncoded[respCol] = strokeDataFrameCleaned[respCol]

# 2.4. Scaling Data.

# Scaling is done to standardize the range of data. This is a necessary 
# aspect of pre-processing while building machine learning algorithms.

# Import 'preprocessing' package from Sklearn.

from sklearn import preprocessing

minMaxScaler = preprocessing.MinMaxScaler().fit(strokeDataFrameEncoded)

matScaled = minMaxScaler.transform(strokeDataFrameEncoded)

strokeDataFrameScaled = strokeDataFrameEncoded.copy()

# Storing the column names from new encoded dataset.

encodedDfCol = list(strokeDataFrameEncoded.columns.values)

# Storing the predictor columns of encoded dataset in a list.

encodedPredColumns = encodedDfCol[:len(encodedDfCol)-1]

for i in range(len(encodedDfCol)-1):
    
    strokeDataFrameScaled[encodedDfCol[i]] = matScaled[:,i]
    
# 3. Model Building

# Once the preprocessing is done, we can go for model building.

# Importing 'numpy' package for using numpy arrays and its operations.

import numpy as np

# Training and test splits
# ------------------------

# Importing the 'train_test_split' library from the 'sklearn' library.
# This is used for splitting the dataset into test and train datasets.

from sklearn.model_selection import train_test_split

# Split of data -> 70% training data means 0.7*number of rows in new dataset.

# Setting the randomness to 50 as it is the last two digits of student number. 

xTrain, xTest, yTrain, yTest = train_test_split(strokeDataFrameScaled[encodedPredColumns]
                                                ,strokeDataFrameScaled[respCol]
                                                ,test_size = 0.3
                                                ,random_state = 50
                                                ,shuffle = True
                                                )

# Importing the 'confusion_matrix', 'accuracy_score', 'f1_score', 
# 'precision_score' and 'recall_score' functions from 'metrics' package for 
# estimating model accuracy.

from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.metrics import f1_score, precision_score, recall_score

# Importing the "MultinomialNB" package for performing Naive Bayes Classification.

from sklearn.naive_bayes import MultinomialNB

# Creating a model for Naive Bayes classifier.

classifierNB = MultinomialNB()

# Training the model.

classifierNB.fit(xTrain,yTrain)

# Storing the prediction of the model.

yPred = np.array(classifierNB.predict(xTest),dtype = 'int64')

yTest = yTest.astype('int64')

# Storing the confusion matrix, accuracy score, f1 score, precision score and 
# recall score of the Naive Bayes Model in separate variables.

confusionMatrixNB = confusion_matrix(yTest,yPred)

accuracyNB = accuracy_score(yTest,yPred)

f1NB = f1_score(yTest,yPred)

precisionNB = precision_score(yTest,yPred,zero_division = 0)

recallNB = recall_score(yTest,yPred)

# Storing the overall results in a pandas dataframe.

overallResults = pd.DataFrame([['Naive Bayes',accuracyNB,f1NB,precisionNB
                                ,recallNB]],columns = ['Model', 'Accuracy'
                                                       , 'F1 Score'
                                                       , 'Precision'
                                                       , 'Recall Score'])

# Importing the "DecisionTreeClassifier" package for using decision tree classifiers.

from sklearn.tree import DecisionTreeClassifier

# Creating a model for decision tree classifiers.

classifierDTC = DecisionTreeClassifier(random_state = 50,criterion = 'entropy')

# Training the model.

classifierDTC.fit(xTrain,yTrain)

# Storing the prediction of the model.

yPred = np.array(classifierDTC.predict(xTest),dtype = 'int64')

# Storing the confusion matrix, accuracy score, f1 score, precision score and 
# recall score of the Decision Tree Model in separate variables.

confusionMatrixDTC = confusion_matrix(yTest,yPred)

accuracyDTC = accuracy_score(yTest,yPred)

f1DTC = f1_score(yTest,yPred)

precisionDTC = precision_score(yTest,yPred,zero_division = 0)

recallDTC = recall_score(yTest,yPred)

# Storing the model results in a pandas dataframe.

modelResults = pd.DataFrame([['Decision Tree',accuracyDTC,f1DTC,precisionDTC
                         ,recallDTC]],columns = ['Model', 'Accuracy'
                                                 , 'F1 Score', 'Precision'
                                                 , 'Recall Score'])
                                                 

overallResults = overallResults.append(modelResults,ignore_index = True)

# Importing the "RandomForestClassifier" package for using random forest (RF) models.

from sklearn.ensemble import RandomForestClassifier

# Creating a model for RF classifier.

classifierRF = RandomForestClassifier(random_state = 50, n_estimators = 1000
                                      ,criterion = 'entropy')

# Training the model.

classifierRF.fit(xTrain,yTrain)

# Storing the prediction of the model.

yPred = np.array(classifierRF.predict(xTest),dtype = 'int64')

# Storing the confusion matrix, accuracy score, f1 score, precision score and 
# recall score of the Random Forest Model in separate variables.

confusionMatrixRF = confusion_matrix(yTest,yPred)

accuracyRF = accuracy_score(yTest,yPred)

f1RF = f1_score(yTest,yPred)

precisionRF = precision_score(yTest,yPred,zero_division = 0)

recallRF = recall_score(yTest,yPred)

# Storing the model results in a pandas dataframe.

modelResults = pd.DataFrame([['Random Forest',accuracyRF,f1RF,precisionRF
                         ,recallRF]],columns = ['Model', 'Accuracy'
                                                 , 'F1 Score', 'Precision'
                                                 , 'Recall Score'])
                                                 
overallResults = overallResults.append(modelResults,ignore_index = True)

# Importing the "SVC" package for using Support Vector Machines (SVM) models.

from sklearn.svm import SVC

# Creating a model for SVC classifier.

classifierSVC = SVC(random_state = 50, kernel = 'sigmoid')

# Training the model.

classifierSVC.fit(xTrain,yTrain)

# Storing the prediction of the model.

yPred = np.array(classifierSVC.predict(xTest),dtype = 'int64')

# Storing the confusion matrix, accuracy score, f1 score, precision score and 
# recall score of the SVC Model in separate variables.

confusionMatrixSVC = confusion_matrix(yTest,yPred)

accuracySVC = accuracy_score(yTest,yPred)

f1SVC = f1_score(yTest,yPred)

precisionSVC = precision_score(yTest,yPred,zero_division = 0)

recallSVC = recall_score(yTest,yPred)

# Storing the model results in a pandas dataframe.

modelResults = pd.DataFrame([['Sigmoid SVC',accuracySVC,f1SVC,precisionSVC
                         ,recallSVC]],columns = ['Model', 'Accuracy'
                                                 , 'F1 Score', 'Precision'
                                                 , 'Recall Score'])
                                                 
overallResults = overallResults.append(modelResults,ignore_index = True)

# We can see that Random Forest model displays the highest accuracy (96.3%). 
# So we can perform cross validation on that model and check the accuracy.

# Importing the 'cross_val_score' package for performing cross validation.

from sklearn.model_selection import cross_val_score

# Performing k = 10 Cross fold validation.

accuracyCVRF = cross_val_score(estimator = classifierRF, X = xTest, y = yTest
                               , cv = 10, n_jobs = 1)

# Storing the Mean and Variance of the accuracies in separate variables.

accMean = str(accuracyCVRF.mean().round(3))
accVar = str((accuracyCVRF.std().round(6))*2)

print('\n')
print("Random Forest Classifier Accuracy: " + accMean + ' +/- ' + accVar)

# There is a minor decrease in accuracy after performing k = 10 CV.

# 4. Hyper parameter tuning.

# The Random Forest model shows the best accuracy among the other models. 
# Hence, we would be tuning the hyper parameters of the model to obtain
# its best accuracy.

# This can be done using the python library 'GridSearchCV'. 
# GridSearchCV implements a "fit" and a "score" method. 
# It also implements "score_samples", "predict", "predict_proba", 
# "decision_function", "transform" and "inverse_transform" if they are 
# implemented in the estimator used. The parameters of the estimator used to 
# apply these methods are optimized by cross-validated grid-search over a 
# parameter grid.

from sklearn.model_selection import GridSearchCV as GSCV

# Tuning Random Forest model with entropy as splitting criterion.

parameterDict1 = {"n_estimators": [10, 100, 150],
                  "bootstrap": [True, False],"criterion": ["entropy"]
                  ,"max_depth": [5, None]
                  }

# Building GridSearch k = 10 Cross fold validation classifier model. 

classifierGS = GSCV(estimator = classifierRF,param_grid = parameterDict1
                    ,scoring = "accuracy",cv = 10,n_jobs = 1)


classifierGS = classifierGS.fit(xTrain,yTrain)

print('\nRandom Forest Best Accuracy: ' + str(classifierGS.best_score_))

print('Random Forest Best Parameters: ' + str(classifierGS.best_params_))

# Predction using Test DataSet.

# Storing the prediction of the model.

yPred = np.array(classifierGS.predict(xTest),dtype = 'int64')

# Storing the confusion matrix, accuracy score, f1 score, precision score and 
# recall score of the GSCV Model in separate variables.

confusionMatrixGSCV = confusion_matrix(yTest,yPred)

accuracyGSCV = accuracy_score(yTest,yPred)

f1GSCV = f1_score(yTest,yPred)

precisionGSCV = precision_score(yTest,yPred,zero_division = 0)

recallGSCV = recall_score(yTest,yPred)

# Storing the model results in a pandas dataframe.

modelResults = pd.DataFrame([['Random Forest n = 10 entropy',accuracyGSCV
                              ,f1GSCV
                              ,precisionGSCV,recallGSCV]]
                            ,columns = ['Model', 'Accuracy'
                                                 , 'F1 Score', 'Precision'
                                                 , 'Recall Score'])
                                                 
overallResults = overallResults.append(modelResults,ignore_index = True)

# Tuning Random Forest model with Gini Indexing as splitting criterion.

parameterDict2 = {"n_estimators": [10, 100, 150],
                  "bootstrap": [True, False],"criterion": ["gini"]
                  ,"max_depth": [5, None]
                  }

# Building GridSearch k = 10 Cross fold validation classifier model. 

classifierGS = GSCV(estimator = classifierRF,param_grid = parameterDict2
                    ,scoring = "accuracy",cv = 10,n_jobs = 1)


classifierGS = classifierGS.fit(xTrain,yTrain)

print('\nRandom Forest Best Accuracy: ' + str(classifierGS.best_score_))

print('Random Forest Best Parameters: ' + str(classifierGS.best_params_))

# Predction using Test DataSet.

# Storing the prediction of the model.

yPred = np.array(classifierGS.predict(xTest),dtype = 'int64')

# Storing the confusion matrix, accuracy score, f1 score, precision score and 
# recall score of the GSCV Model in separate variables.

confusionMatrixGSCV = confusion_matrix(yTest,yPred)

accuracyGSCV = accuracy_score(yTest,yPred)

f1GSCV = f1_score(yTest,yPred)

precisionGSCV = precision_score(yTest,yPred,zero_division = 0)

recallGSCV = recall_score(yTest,yPred)

# Storing the model results in a pandas dataframe.

modelResults = pd.DataFrame([['Random Forest n = 10 Gini Indexing'
                              ,accuracyGSCV,f1GSCV,precisionGSCV,recallGSCV]]
                            ,columns = ['Model', 'Accuracy'
                                                 , 'F1 Score', 'Precision'
                                                 , 'Recall Score'])
                                                 
overallResults = overallResults.append(modelResults,ignore_index = True)

# There is a slight increase in model accuracy due to hyper parameter tuning.

# Creating the Heat Map Plot for the Confusion Matrix.

cmDataFrame = pd.DataFrame(confusionMatrixGSCV, index = (0, 1), columns = (0, 1))

sns.heatmap(cmDataFrame,cmap = sns.diverging_palette(250, 15, as_cmap = True))

print("Test Data Accuracy: " + str(accuracyGSCV.round(4)))
