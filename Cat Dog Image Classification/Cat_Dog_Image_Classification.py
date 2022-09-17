# Storing the file path of the dataset that is comprised of images of dogs and  
# cats in separate variables for better readability.

filePath1 = 'C:/Users/prash/Downloads/MSc Data Science and Analytics (Sem 2)/'

filePath2 = 'Applied Machine Learning/Project 3/'

# Saving the overall project directory in a string variable.

projectDir = filePath1 + filePath2

# Importing the 'os' package for file handling operations.

import os

# Changing the working directory to the dataset path.

os.chdir(projectDir)

# Importing the 'cv2' package for working with images.

import cv2

# The 'glob' package is used for retrieving pathnames matching a specified
# pattern according to rules related to Unix Shell. 

import glob

# Importing 'numpy' package for using numpy arrays and its operations.

import numpy as np

# Since there would be some visualisation involved, we would be using the 
# "matplotlib.pyplot" and "seaborn" packages.

import matplotlib.pyplot as plt

import seaborn as sns

# Storing the file paths of the train and test images in a list.

trainTestFileList = ['Cat Dog images/train/*','Cat Dog images/test/*']

# Part A: Pre-processing Phase:
    
# As part of pre-processing we will read the images, re-size them to 350*350*3
# and normalize them at last.

# Importing 're' package for using regular expressions.

import re

# Importing 'pathlib' package for using file path operations.

from pathlib import Path 

# Creating a function that takes the input file path and returns the sorted 
# filenames based on the numeric value present in the filename string.

def sortFileNameNumeric(file):
    return int(re.compile(r'.*?(\d+).*?').match(Path(file).name).groups()[0])

# Creating a function that takes the input file path and returns the list
# of resized images, normalized images and the image labels from the files 
# present.

def genImageListandLabels(filePath):

    # Creating a list for storing the images for image labels.
    
    listImages = []
    
    listImageLabels = []
    
    listNormImages = []
    
    # Tarversing through each image.
    
    fileNames = sorted(glob.glob(filePath), key = sortFileNameNumeric)
    
    for fileName in fileNames:
        
        # Reading the image from the directory.
        
        image = cv2.imread(fileName)
    
        # Resizing the image to 350, 350, 3.
        
        imageReSize = cv2.resize(image, (350,350))
        
        # Normalizing the image.
        
        normImageReSize = cv2.normalize(imageReSize,np.zeros(shape = (350,350))
                                        ,0, 255, norm_type = cv2.NORM_MINMAX
                                        , dtype = cv2.CV_32F)
        
        # Appending the image to the image list.
        
        listImages.append(imageReSize)
        
        # Appending the normalized image to the image list.
        
        listNormImages.append(normImageReSize)
        
        # Creating the labels for the class of image. 
        # '0' stands for cat and '1' stands for dog.
        # For training data, the labels would be extracted from the filenames.
        # However, there are no labels in the testing data and arbitrary
        # labels (0) would be assigned to it. These values would change once
        # the model building is done.
        
        imageLabel = 0
        
        if fileName.split('/')[1].split('\\')[1].split('.')[0] == 'dog':
            
            imageLabel = 1
            
        listImageLabels.append(imageLabel)   
        
    return (listImages,listNormImages,listImageLabels)

# Storing the Training Data in a list.

xTrain = xTrainNorm = xTest = xTestNorm = yTrain = yTest = []

xTrain,xTrainNorm,yTrain = genImageListandLabels(trainTestFileList[0])

# Storing the Testing Data in a list.

xTest,xTestNorm,yTest = genImageListandLabels(trainTestFileList[1])

# Coverting the list of 3D images to 4D Arrays.

xTrain = np.stack(xTrain, axis = 0)

xTrainNorm = np.stack(xTrainNorm, axis = 0)

yTrain = np.stack(yTrain, axis = 0)

xTestNorm = np.stack(xTestNorm, axis = 0)

# Converting the Labels to 1D array.

xTest = np.stack(xTest, axis = 0)

yTest = np.stack(yTest, axis = 0)

print('\nDimensions of the training data:',xTrain.shape) 

print('\nDimensions of the normalized training data:',xTrainNorm.shape) 

print('\nDimensions of the training labels:',yTrain.shape) 

print('\nDimensions of the testing data:',xTest.shape) 

print('\nDimensions of the normalized testing data:',xTestNorm.shape) 

print('\nDimensions of the testing labels:',yTest.shape) 

# Creating a function for visualizing the samples of the training images.

def visImage(inputImage, inputLabel,isTrainTest):
    
    for y, label in enumerate(np.unique(inputLabel)):
        
        indexes = np.flatnonzero(inputLabel == y)
        
        indexes = indexes[0:4]
    
        for i, j in enumerate(indexes):
                   
            plt.subplot(4,len(np.unique(inputLabel))
                        ,y + 1 + i * len(np.unique(inputLabel)))
            
            plt.imshow(inputImage[j].astype('uint8'))
            
            plt.axis('off')
            
            # If the training dataset is passed, the labels will be assigned
            # in accordance to the animal name. Since, prediction has not
            # yet been done for the testing dataset,a default label 'Animal'
            # would be used.
                        
            strLabel = 'Animal'
            
            if isTrainTest:
                
                # Assigning labels for cats and dogs. 
                # '0' stands for cat and '1' stands for dog.
                
                strLabel = 'Cat'
                
                if label == 1:
                    
                    strLabel = 'Dog'
                    
            if i == 0:
                
                plt.title(strLabel)
                
    plt.show()

# Visualizing the train images before and after normalization.

visImage(xTrain,yTrain,True)

visImage(xTrainNorm,yTrain,True)

# Visualizing the test images before and after normalization.

visImage(xTest,yTest,False)

visImage(xTestNorm,yTest,False)

# Reshaping the 4D images to 2D data for analysis.

xTrain2D = np.reshape(xTrain, (xTrain.shape[0], -1))

xTrainNorm2D = np.reshape(xTrainNorm, (xTrainNorm.shape[0], -1))

xTest2D = np.reshape(xTest, (xTest.shape[0], -1))

xTestNorm2D = np.reshape(xTestNorm, (xTestNorm.shape[0], -1))

print('\nDimensions of the training data:',xTrain.shape) 

print('\nDimensions of the 2D training data:',xTrain2D.shape) 

print('\nDimensions of the normalized training data:',xTrainNorm.shape) 

print('\nDimensions of the 2D normalized training data:',xTrainNorm2D.shape) 

print('\nDimensions of the training labels:',yTrain.shape) 

print('\nDimensions of the testing data:',xTest.shape) 

print('\nDimensions of the 2D testing data:',xTest2D.shape) 

print('\nDimensions of the normalized testing data:',xTestNorm.shape) 

print('\nDimensions of the 2D normalized testing data:',xTestNorm2D.shape) 

print('\nDimensions of the testing labels:',yTest.shape) 

# Principal component analysis (PCA) is one of the dimension reduction
# techniques. It is derived from the eigenvalues and eigenvectors of correlation 
# matrix of a dataset. The eigenvector Matrix is multiplied with the input data
# to get the principal components. Before that, we need to check if PCA is 
# needed or not by using Bartlett's test of sphericity.

# Bartlett's test of sphericity:
# H0 = Correlation Matrix is an identity matrix which means variables are uncorrelated.
# HA = Correlation Matrix is not an identity matrix which means variables are 
# correlated and suitable for factor analysis.

# PCA is a special case of factor analysis.

# Importing 'calculate_bartlett_sphericity' function from 'factor_analyzer'.

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

# Since there are too many observations in the normalized training data,  
# we take the first ten rows and columns for the Bartlett's test.

statistic, p_value = calculate_bartlett_sphericity(xTrainNorm2D[1:10,1:10])

s = '\nSince p_value (' + str(p_value) + ') > 0.05, we fail to reject H0.'

if p_value < 0.05:
    
    s = '\nSince p_value (' + str(p_value) + ') <= 0.05, we reject H0.'
    
print(s)

# Since p_value (2.1035549803098057e-47) <= 0.05, we reject H0.
# This means that there is no sufficient evidence to show that the Correlation 
# Matrix is an identity matrix.

# Importing 'PCA' function from 'sklearn'.

from sklearn.decomposition import PCA

# In PCA we take the first two components since they hold the highest variance.

xTrainNorm2DPCA = PCA(2).fit_transform(xTrainNorm2D)

xTestNorm2DPCA = PCA(2).fit_transform(xTestNorm2D)

print('\nDimensions of the training data:',xTrain.shape)

print('\nDimensions of the 2D training data:',xTrain2D.shape)

print('\nDimensions of the normalized training data:',xTrainNorm.shape)

print('\nDimensions of the 2D normalized training data:',xTrainNorm2D.shape)

print('\nDimensions of the PCA normalized training data:',xTrainNorm2DPCA.shape)

print('\nDimensions of the training labels:',yTrain.shape)

print('\nDimensions of the testing data:',xTest.shape)

print('\nDimensions of the 2D testing data:',xTest2D.shape)

print('\nDimensions of the normalized testing data:',xTestNorm.shape)

print('\nDimensions of the 2D normalized testing data:',xTestNorm2D.shape)

print('\nDimensions of the PCA normalized testing data:',xTestNorm2DPCA.shape)

print('\nDimensions of the testing labels:',yTest.shape)

# We conclude the pre-processing with the PCA. Next, we go for model building.

# Part B: Training phase:
    
# Importing the 'pandas' package for working with dataframes.

import pandas as pd

# Since this is a binary classification problem, we could use algorithms such
# as Logistic Regression (LR), Decision Tree Classifier (DTC), Random Forests 
# (RF) and Stochastic Gradient Boosting Models (GBM).

# Importing the 'train_test_split' library from the 'sklearn' library.
# Since the test dataset does not contain any prediction values, we need to 
# split the training dataset into train and validation datasets.

from sklearn.model_selection import train_test_split

# Split of data -> 80% training data means 0.8*number of rows in new dataset.

# Setting the randomness to 150. 

trainX, valX, trainY, valY = train_test_split(xTrainNorm2DPCA
                                              ,yTrain
                                              ,test_size = 0.2
                                              ,random_state = 150
                                              ,shuffle = True
                                              )

pd.Series(yTrain).value_counts()

# The baseline accuracy is 50%. Our model's accuracy must be higher than this.

# Importing the 'confusion_matrix', 'accuracy_score' functions from 
# 'metrics' package for estimating model accuracy.

from sklearn.metrics import confusion_matrix, accuracy_score 

# Importing the "LogisticRegression" package for performing Logistic Regression.

from sklearn.linear_model import LogisticRegression

# Creating a model for decision tree classifiers.

LR = LogisticRegression(random_state = 150)

# Training the model.

LR.fit(trainX,trainY)

# Storing the prediction of the model.

yEst = np.array(LR.predict(valX),dtype = 'int64')

# Storing the confusion matrix and accuracy score of the Logistic Regression
# Model in separate variables.

cmLR = confusion_matrix(valY,yEst)

accLR = accuracy_score(valY,yEst)

# Storing the model results in a pandas dataframe.

allResults = pd.DataFrame([['Logistic Regression',accLR]],columns = ['Model', 'Accuracy'])

# Importing the "DecisionTreeClassifier" package for using decision tree classifiers.

from sklearn.tree import DecisionTreeClassifier

# Creating a model for decision tree classifiers.

DTC = DecisionTreeClassifier(random_state = 150,criterion = 'entropy')

# Training the model.

DTC.fit(trainX,trainY)

# Storing the prediction of the model.

yEst = np.array(DTC.predict(valX),dtype = 'int64')

# Storing the confusion matrix and accuracy score of the Decision Tree Model 
# in separate variables.

cmDTC = confusion_matrix(valY,yEst)

accDTC = accuracy_score(valY,yEst)

# Storing the model results in a pandas dataframe.

modelResult = pd.DataFrame([['Decision Tree',accDTC]],columns = ['Model', 'Accuracy'])

allResults = allResults.append(modelResult,ignore_index = True)

# Importing the "RandomForestClassifier" package for using random forest (RF) models.

from sklearn.ensemble import RandomForestClassifier

# Creating a model for RF classifier.

RF = RandomForestClassifier(random_state = 150, n_estimators = 2000,criterion = 'entropy')

# Training the model.

RF.fit(trainX,trainY)

# Storing the prediction of the model.

yEst = np.array(RF.predict(valX),dtype = 'int64')

# Storing the confusion matrix and accuracy score of the Random Forest Model 
# in separate variables.

cmRF = confusion_matrix(valY,yEst)

accRF = accuracy_score(valY,yEst)

# Storing the model results in a pandas dataframe.

modelResult = pd.DataFrame([['Random Forest',accRF]],columns = ['Model', 'Accuracy'])
                                                 
allResults = allResults.append(modelResult,ignore_index = True)

# Importing the "GradientBoostingClassifier" package for using stochastic
# gradient boosting models (GBM).

from sklearn.ensemble import GradientBoostingClassifier

# Creating a model for decision tree classifiers.

GBM = GradientBoostingClassifier(n_estimators = 300, learning_rate = 0.05
                                 ,max_depth = 4, random_state = 150)

# Training the model.

GBM.fit(trainX,trainY)

# Storing the prediction of the model.

yEst = np.array(GBM.predict(valX),dtype = 'int64')

# Storing the confusion matrix and accuracy score of the GBM in separate variables.

cmGBM = confusion_matrix(valY,yEst)

accGBM = accuracy_score(valY,yEst)

# Storing the model results in a pandas dataframe.

modelResult = pd.DataFrame([['Stochastic Gradient Boosting',accGBM]],columns = ['Model', 'Accuracy'])

allResults = allResults.append(modelResult,ignore_index = True)

print(allResults)

# Plotting the results using a bar plot.

plt.figure()

allResults.plot.bar(x = 'Model', y = 'Accuracy',title = "Bar plot of Model vs Accuracy",rot = 30)

plt.xlabel("Model Name")

plt.show()

# Random Forest Model has the best accuracy out of all the models. 
# Hence, it will be used for predicting the testing data.

# Training the model.

RF.fit(xTrainNorm2DPCA,yTrain)

# Storing the prediction of the model.

yFinalEst = np.array(RF.predict(xTestNorm2DPCA),dtype = 'int64')

# Once the prediction is down, we go for hyperparameter tuning of the best model.

# Part C: Optimization phase:
    
# Hyperparameter tuning can be done using the python library 'GridSearchCV'. 

from sklearn.model_selection import GridSearchCV as GSCV

# Tuning Random Forest model with entropy as splitting criterion.

paramDict1 = {"n_estimators": [25, 250, 350],"bootstrap": [True, False]
              ,"criterion": ["entropy"],"max_depth": [5, 7, None]}

# Building GridSearch k = 10 Cross fold validation classifier model. 

GS = GSCV(estimator = RF,param_grid = paramDict1,scoring = "roc_auc",cv = 10,n_jobs = 1)

GS = GS.fit(trainX,trainY)

print('\nRandom Forest Best Accuracy: ' + str(GS.best_score_))

print('\nRandom Forest Best Parameters: ' + str(GS.best_params_))

# Prediction using Validation DataSet.

yEst = np.array(GS.predict(valX),dtype = 'int64')

# Storing the confusion matrix and accuracy score of the GSCV Model in separate variables.

cmGSCV = confusion_matrix(valY,yEst)

accGSCV = accuracy_score(valY,yEst)

# Storing the model results in a pandas dataframe.

modelResult = pd.DataFrame([['Random Forest n = 10 entropy',accGSCV]],columns = ['Model', 'Accuracy'])
                                                 
allResults = allResults.append(modelResult,ignore_index = True)

# Tuning Random Forest model with Gini Indexing as splitting criterion.

paramDict2 = {"n_estimators": [25, 250, 350],"bootstrap": [True, False]
              ,"criterion": ["gini"],"max_depth": [5, 7, None]}

# Building GridSearch k = 10 Cross fold validation classifier model. 

GS = GSCV(estimator = RF,param_grid = paramDict2,scoring = "roc_auc",cv = 10,n_jobs = 1)

GS = GS.fit(trainX,trainY)

print('\nRandom Forest Best Accuracy: ' + str(GS.best_score_))

print('\nRandom Forest Best Parameters: ' + str(GS.best_params_))

# Prediction using Validation DataSet.

yEst = np.array(GS.predict(valX),dtype = 'int64')

# Storing the confusion matrix and accuracy score of the GSCV Model in separate variables.

cmGSCV = confusion_matrix(valY,yEst)

accGSCV = accuracy_score(valY,yEst)

# Storing the model results in a pandas dataframe.

modelResult = pd.DataFrame([['Random Forest n = 10 Gini Indexing',accGSCV]],columns = ['Model', 'Accuracy'])
                                                 
allResults = allResults.append(modelResult,ignore_index = True)

# Plotting the results using a bar plot.

plt.figure()

# Storing the plot title in a string variable.

plotTitle = "Bar plot of Model vs Accuracy after hyperparameter tuning"

plt.ylabel("Model Name")

plt.xlabel("Accuracy")

plt.barh(allResults['Model'], allResults['Accuracy'])

plt.show()

# We can see a slight increase in accuracy after hyper parameter tuning.

# Creating the Heat Map Plot for the Confusion Matrix.

cmDataFrame = pd.DataFrame(cmGSCV, index = (0, 1), columns = (0, 1))

sns.heatmap(cmDataFrame,cmap = sns.diverging_palette(250, 12, as_cmap = True,center = "dark"))

print("Test Data Accuracy: " + str(accGSCV.round(4)))
