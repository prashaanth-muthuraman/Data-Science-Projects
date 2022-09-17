# Preprocessing
# -------------

# The 'glob' package is used for retrieving pathnames matching a specified
# pattern according to rules related to Unix Shell. 

import glob

# Storing the file paths of the Ham and Spam data in separate variables for 
# better readability.

filePath1 = 'C:/Users/prash/Downloads/MSc Data Science and Analytics (Sem 2)/'

filePath2 = 'Applied Machine Learning/Project 1/'

hamSpamFileList = ['enron1.tar/ham/*','enron1.tar/spam/*']

hamFilePath = filePath1 + filePath2 + hamSpamFileList[0]

spamFilePath = filePath1 + filePath2 + hamSpamFileList[1]

# Creating a function that takes the input file path and returns the list
# of cleaned emails from the files present.

# For reading Spam emails 'Latin-1' encoding should be used.

# Here, cleaning refers to handling duplicate and empty emails.
# If a file is either fully empty or contains an empty email i.e. an email 
# with just the subject but no content, then that data will not be processed.
# However, if a file contains duplicate emails, then that data will be 
# processed after deduplication.

def fileClean(filePath,isHamOrSpam = True):
    
    # Creating a list for storing the cleaned emails. 

    classList = []
    
    # Storing the default encoding in a string variable.
    
    strEncoding = 'utf-8'
    
    # Checking if the input is Spam or Ham.
    
    # For opening 
    
    if ~isHamOrSpam:
        
        strEncoding = 'Latin-1'
        
    # The glob function returns a list of paths matching a pathname pattern.

    for fileName in glob.glob(filePath):
        
        with open(fileName,encoding = strEncoding) as f:

            # The contents of the email file are assumed to be entered in a 
            # line-by-line format and hence, read stored in a separate list.
            # They are lowercased as part of pre-processing.
            
            readFileLine =  f.read().lower().split('\n')
            
            # Checking if the file is fully empty.
            
            if len(readFileLine) > 0:
                
                # It is assumed that the first line of a mail always starts 
                # with a subject, and the content follows from the second line.
            
                # Storing the indexes of the word 'subject' present in each 
                # list value.
            
                subIndexList = [i for i in range(len(readFileLine)) if readFileLine[i].startswith('subject')]
                
                # If all the lines are either blank or holds only whitespaces,
                # then that file is considered to be an empty file.
                    
                # If the file is not an empty file, then the duplicate emails  
                # should be removed.
                
                # Creating a list for storing unique emails.
                
                uniqEmailList = []
    
                for i in range(len(subIndexList)):
                    
                    # Storing the content of the email (excluding subject and '\n')
                    # in a string variable.
                    
                    mailStr1 = ''.join(readFileLine[subIndexList[i]+1:len(readFileLine)])
                    
                    # Storing the entire email content (excluding '\n') in a list.
                    
                    tempList = [] + readFileLine[subIndexList[i]:len(readFileLine)]
                    
                    if i != len(subIndexList) - 1: 
                        mailStr1 = ''.join(readFileLine[subIndexList[i]+1:subIndexList[i+1]])
                        tempList = [] + readFileLine[subIndexList[i]:subIndexList[i+1]]
                    
                    # Creating a new string variable for storing the original email
                    # string.
                    
                    mailStr2 = ''
                    
                    for j in tempList:
                        mailStr2 += j + '\n'
                    
                    # Checking if the email is neither empty nor repetitive.
                    
                    if mailStr1.strip() != '' and mailStr2.rstrip() +'\n' not in uniqEmailList:
                        uniqEmailList.append(mailStr2.rstrip()+'\n')
                        
                # Storing the final and cleaned emails in the Class List.
                
                if len(uniqEmailList) > 0:
                    classList.append(''.join(uniqEmailList))

    return classList

# Storing the Ham Data in a list.

hamList = [] + fileClean(hamFilePath)

# Storing the Spam Data in a list.

spamList = [] + fileClean(spamFilePath,False)

# Creating separate dataframes for Ham and Spam data from the respective lists.
# To do this we need to import the "pandas" package.

import pandas as pd

hamDataFrame = pd.DataFrame(hamList,columns = ['Emails'])

spamDataFrame = pd.DataFrame(spamList,columns = ['Emails'])

# Creating a new column 'target' to identify the mail as ham or spam.
# If the 'target' value is 1, then it is ham. Else, it is spam. 

hamDataFrame['target'] = 1

spamDataFrame['target'] = 0

# Training and test splits
# ------------------------

# Importing the 'train_test_split' library from the 'sklearn' library.
# This is used for splitting the dataset into test and train datasets.

from sklearn.model_selection import train_test_split

# Saving all the spam and ham emails in a single dataframe.

allEmailDataFrame = pd.concat([spamDataFrame,hamDataFrame])

# Split of data -> 70% training data means 0.7*number of rows in new dataset.

# Setting the randomness to 50 as it is the last two digits of student number. 

xTrain, xTest, yTrain, yTest = train_test_split(allEmailDataFrame['Emails']
                                                ,allEmailDataFrame['target']
                                                ,test_size = 0.3
                                                ,random_state = 100
                                                ,shuffle = True
                                                )

# Converting the training and testing data into pandas dataframes.

trainDataFrame = pd.concat([xTrain,yTrain],axis = 1)

testDataFrame = pd.concat([xTest,yTest],axis = 1)

# Saving the Training and Testing data into Excel Files.

trainDataFrame.to_excel(filePath1 + filePath2 +'Train_Data.xlsx','Train')

testDataFrame.to_excel(filePath1 + filePath2 +'Test_Data.xlsx','Test')

# Collecting statistics on the resulting training and test sets such as 
# total numbers of spam and non-spam emails in each set.

# Storing the ham and spam email counts in a dataframe.

dfTarget = allEmailDataFrame.groupby('target',as_index = False).count()

# Storing the ham and spam email counts in separate variables.

spamCount = int(dfTarget[dfTarget['target'] == 0]['Emails'])

hamCount = int(dfTarget[dfTarget['target'] == 1]['Emails'])

print('The total number of spam emails is ' + str(spamCount))

print('\nThe total number of ham emails is ' + str(hamCount))

# Creating a function to calculate the total numbers of spam and ham emails.

def calculateEmailCount(inputDf):
    
    # Storing the unique classes such as ham and spam in a dataframe.
    
    uniqueClasses = pd.unique(inputDf['target'])
    
    # Creating a dictionary to store the spam and ham email count.
    
    spamHamDict = {}
    
    for uniqueClass in uniqueClasses:
        
        strHamorSpam = 'Ham'

        if uniqueClass == 0:
            
            strHamorSpam = 'Spam'
        
        # Storing the emails of a class in a variable.
        
        emailCollection = inputDf[inputDf['target'] == uniqueClass]['Emails']
                
        # Every email starts with the word 'subject' and hence we count the
        # number of occurrences of that word in each email.
        
        cnt = 0
        
        for email in emailCollection:
            
            cnt += str(email).count('subject:') + str(email).count('subject :')
        
        spamHamDict[strHamorSpam] = cnt
        
    return spamHamDict

# Saving the total number of spam and ham emails present in each dataset.

trainSpamHamDict = calculateEmailCount(trainDataFrame)

testSpamHamDict = calculateEmailCount(testDataFrame)

# Feature extraction
# ------------------

# Before feature extraction, we need to peform text processing such as removal
# of stop words, lemmatization, etc.

# Importing 'nltk' package for text processing.
        
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

# Adding new stop words that could be present in a mail.

newStopWords = ('subject','subject:','subject: ','best regards','best','regards'
                ,'looking forward','forward','looking','email','email address'
                ,'re', 're:','re :', 'to:','to :','from', 'from:', 'from :'
                ,'fw', 'fw:', 'fw :','_','cc:','cc','cc :','bcc:','bcc :','bcc'                
                ) 

# Saving the overall collection of stop words in a set variable.

stopWords = stopWords.union(newStopWords)

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

# Importing 're' pakcage for using regular expressions.

import re

# Creating a function that takes the input dataframe and returns the dataframe
# after performing text processing such as stop word removal, lemmatization.

def textProcessing(inputDf2):
    
    # Creating a copy of the input Dataframe.
    
    inputDf = inputDf2.copy()
    
    # Storing the emails in a variable.
    
    emailCollection = inputDf['Emails']
        
    # Creating a list for storing the strings after text processing.
    
    finalWords = []
    
    for email in emailCollection:
        
        # Creating a list for storing the strings after stop word removal.
    
        filteredWords = []
        
        # Cleaning the email string.
        
        cleanEmail = re.sub(r"[\;\?\<\>\,\"\|\:\.\`\~\{\}\\\/@$\%\[\]\(\)\^\-\+\&#\*\!\_\=]*", "", email)
        
        for word in word_tokenize(cleanEmail):
            
            if word not in stopWords:
                
                filteredWords.append(word)
                
        lemmatizer = WordNetLemmatizer()
    
        # Lemmatization of words with adverb, verb and noun.
        
        str1 = lemmatizer.lemmatize(" ".join(filteredWords), pos = "n")
        str2 = lemmatizer.lemmatize(str1, pos = "v")
        str3 = lemmatizer.lemmatize(str2, pos = "a")
        
        finalWords.append(str3)
    
    inputDf['Emails'] = finalWords
    
    return inputDf

# Storing the dataframes after text processing.

trainDataFrameAfterProc = textProcessing(trainDataFrame)

testDataFrameAfterProc = textProcessing(testDataFrame)

# Importing the "TfidfVectorizer" package for TF-IDF encoding.

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

# Creating a function that takes the input dataframe and returns the TF-IDF
# vector.

def tfIdfTransform(inputDf2):
    
    # Creating a copy of the input Dataframe.
    
    inputDf = inputDf2.copy()
    
    # Storing the unique classes such as ham and spam in a dataframe.
    
    uniqueClasses = pd.unique(inputDf['target'])
    
    # Creating a dictionary to store the spam and ham email transforms.
    
    spamHamTransformDict = {}
    
    # Creating a dictionary to store the spam and ham vocabularies.
    
    spamHamVocabDict = {}

    for uniqueClass in uniqueClasses:
        
        strHamorSpam = 'Ham'

        if uniqueClass == 0:
            
            strHamorSpam = 'Spam'
        
        # Storing the emails of a class in a variable.
        
        emailCollection = inputDf[inputDf['target'] == uniqueClass]['Emails']
        
        # Storing all the texts in a variable.
        
        text = ' '.join(emailCollection)
        
        # Creating the TF-IDF transform.
        
        vectorizer = TfidfVectorizer()
        
        # Tokenizing and building vocabulary.
        
        vectorizer.fit(emailCollection)
        
        # Encoding the document.
        
        vector = vectorizer.transform([text])

        spamHamTransformDict[strHamorSpam] = vector.toarray()
        
        spamHamVocabDict[strHamorSpam] = vectorizer.vocabulary_
    
    return spamHamTransformDict, spamHamVocabDict

# Saving the TF-IDF Encoded data for the train dataset.

trainDataTfIdfVector = tfIdfTransform(trainDataFrameAfterProc)[0]

trainDataTfIdfVocab = tfIdfTransform(trainDataFrameAfterProc)[1]

# Exploratory data analysis
# -------------------------

# Finding the top-20 most frequently used words in spam and non-spam emails 
# and using a bar plot to show their relative frequencies.

# Since there would be some visualisation involved, we would be using the 
# "matplotlib.pyplot" and "seaborn" packages.

import matplotlib.pyplot as plt

import seaborn as sns

# Creating a dictionary to hold the top-20 most frequently used words in spam 
# and non-spam emails.

trainDataVocab20 = {}

for vocabKey in trainDataTfIdfVocab.keys():
    
    # Storing the sub dictionary for individual main keys in a variable.
    
    vocabSubDict = trainDataTfIdfVocab[vocabKey]
    
    # Creating a dictionary to hold the top-20 most frequently used words in  
    # each class of email.
    
    classDict = {}
    
    for i in range(20):
        
        # Storing the subvalue in a variable.
        
        val = sorted(vocabSubDict.values(),reverse = True)[i]
        
        key = list(vocabSubDict.keys())[list(vocabSubDict.values()).index(val)]
        
        classDict[key] = val
        
    trainDataVocab20[vocabKey] = classDict
           
    plt.figure()
    
    # Storing the label for the x-axis in a string variable.
    
    xlabel = 'top-20 most frequently used words in ' + vocabKey + ' emails'
    
    # Storing the title of the bar plot in a string variable.
    
    title = 'Bar plot of the ' + xlabel
    
    # Creating a dataframe for plotting.
    
    dfPlot = pd.DataFrame(classDict.keys(),columns = ['Keys'])
    
    dfPlot['Values'] = [i/sum(classDict.values()) for i in classDict.values()]
    
    # Generating the bar plot.
    
    plot1 = sns.barplot(x = "Keys", y = "Values", data = dfPlot)
    
    # Rotating the x-axis labels by 90 degrees.
    
    plot1.set_xticklabels(plot1.get_xticklabels(),rotation = 90)
    
    # Adding the xlabel to the plot.
    
    plot1.set_xlabel(xlabel)
    
    # Adding the ylabel to the plot.
    
    plot1.set_ylabel('Frequency')
    
    # Displaying the plot.
    
    plt.show()

# Comparing the distribution of email lengths in spam and non-spam emails 
# using boxplot.

# Creating a function to calculate the lengths of spam and non-spam emails.

def calculateEmailLength(inputDf):
    
    # Storing the unique classes such as ham and spam in a dataframe.
    
    uniqueClasses = pd.unique(inputDf['target'])
    
    # Creating dataframes to store the spam and ham email lengths.
    
    hamEmailDataFrame = pd.DataFrame()
    
    spamEmailDataFrame = pd.DataFrame()

    for uniqueClass in uniqueClasses:
                
        # Storing the emails of a class in a variable.
        
        emailCollection = inputDf[inputDf['target'] == uniqueClass]['Emails']
                
        # Storing the length of each email in a list variable.
        
        emailLengthList = []
        
        for email in emailCollection:
            
            emailLengthList.append(len(email))
        
        if uniqueClass == 0:
            
            hamEmailDataFrame = pd.DataFrame(emailLengthList
                                             ,columns = ['Lengths']
                                             )
            
        else:
            
            spamEmailDataFrame = pd.DataFrame(emailLengthList
                                              ,columns = ['Lengths']
                                              )
        
    # Creating a new column 'target' to identify the mail as ham or spam.
    # If the 'target' value is 1, then it is ham. Else, it is spam. 
    
    hamEmailDataFrame['Target'] = 'Ham'
    
    spamEmailDataFrame['Target'] = 'Spam'

    return pd.concat([spamEmailDataFrame,hamEmailDataFrame])

# Storing the spam and ham email lengths of training dataset.

trainSpamHamEmailLengthDf = calculateEmailLength(trainDataFrame)

plt.figure()

# Generating the box plot.

plot1 = sns.boxplot(x = trainSpamHamEmailLengthDf['Target']
                    ,y = trainSpamHamEmailLengthDf['Lengths']
                    )

# Adding title to the plot.

plt.title('Boxplot of spam and ham email lengths')

# Adding the xlabel to the plot.

plot1.set_xlabel('Email Type')

# Adding the ylabel to the plot.

plot1.set_ylabel('Email Length')

# Displaying the plot.

plt.show()

# The boxplot highlights the distribution of ham and spam emails. 
# The spam email lengths are more distributed than ham emails. 
# There are many outliers present in the Spam emails than the Ham emails. 
# This is because spam emails contain too much unnecessary and irrelevant 
# information. Sometimes spam emails have fancy catchphrases which grabs 
# people’s attention. This distribution would not change much even if the words 
# are optimally filtered and tokenized.

# Plotting the histogram of the TF-IDF Encoded data for Spam and Ham.

for vectorKey in trainDataTfIdfVector.keys():
    
    # Storing the TF_IDF data for a email type in a variable.
    
    tfIdfArray = trainDataTfIdfVector[vectorKey][0]
    
    plt.figure()

    # Generating the histogram. Increasing the scale of the input by 10 for 
    # better readability.
    
    plot1 = sns.distplot(tfIdfArray*10,bins = 5)
    
    # Adding title to the plot.
    
    plt.title('Histogram of the TF-IDF Encoded data for ' + vectorKey)
    
    # Adding the xlabel to the plot.
    
    plot1.set_xlabel('TF-IDF Encoded data for ' + vectorKey)
    
    # Adding the ylabel to the plot.
    
    plot1.set_ylabel('Frequency')
    
    # Displaying the plot.
    
    plt.show()
    
# The distirbution of TF-IDF encoding is skewed to the right from normal
# distribution. This means mean < median < mode.

# Supervised classification
# -------------------------

# Before performing cross validation, the model must be trained manually. 
# k-cross fold validation is the process in which the training data is randomly 
# split into 'k' folds without replacement, where 'k-1' folds are used for model
# training and one-fold is used for testing. This algorithm is applied k-times
# to obtain 'k' models and performance estimates. The models are independent
# to each other and hence, the average performance is calculated to obtain a 
# performance estimate. This is extremely reliable while processing unseen data
# and balances the variance-bias trade off.

# Importing 'numpy' package for using numpy arrays and its operations.

import numpy as np

# Importing the 'cross_val_score' package for performing cross validation.

from sklearn.model_selection import cross_val_score

# Importing the 'accuracy_score' package for estimating model accuracy.

from sklearn.metrics import accuracy_score

# Importing the 'TfidfTransformer' package for performing TF-IDF transforms.

from sklearn.feature_extraction.text import TfidfTransformer

# Importing the "MultinomialNB" package for performing Naive Bayes Classification.

from sklearn.naive_bayes import MultinomialNB

# Importing the "Pipeline" package for building models and re-using them.

from sklearn.pipeline import Pipeline

# Creating a pipeline for Naive Bayes classifier.

textClfNB = Pipeline([('vect', TfidfVectorizer())
                      ,('tfidf', TfidfTransformer())
                      ,('clf', MultinomialNB()),
                      ])

textClfNB.fit(xTrain,yTrain)

# Storing the accuracy of the Naive Bayes Model in a variable.

accuracyNB = accuracy_score(yTest,textClfNB.predict(xTest))*100

print('\nThe accuracy for the Naive Bayes classifier is ' + str(accuracyNB))

# k = 10 is the most used value for small sample sizes. Using a large 
# 'k' value will consume more training data at each iteration and will result
# in low bias while estimating the model performance. However, computation
# complexity will increase along with the overall error for every high value.

# Storing the scores of k = 10 cross fold validation on Naive Bayes Model.

scoresCVNB = cross_val_score(estimator = textClfNB, X = xTrain, y = yTrain
                             , cv = 10, n_jobs = 1)

# Storing the accuracy and standard deviation in variables.

accuracyCVNB = np.mean(scoresCVNB)*100

print('\nThe accuracy for k = 10 cross fold validation on Naive Bayes Model'
      ,' is ' + str(accuracyCVNB),sep = '')

stdCVNB = np.std(scoresCVNB)

print('\nThe standard deviation for k = 10 cross fold validation on Naive'
      ' Bayes Model',' is ' + str(stdCVNB),sep = '')

# Importing the "DecisionTreeClassifier" package for using decision tree classifiers.

from sklearn.tree import DecisionTreeClassifier

# Creating a pipeline for decision tree classifiers.

textClfDTC = Pipeline([('vect', TfidfVectorizer())
                      ,('tfidf', TfidfTransformer()),
                      ('clf', DecisionTreeClassifier()),]
                     )

textClfDTC.fit(xTrain,yTrain)

# Storing the accuracy of the decision tree classifier model in a variable.

accuracyDTC = accuracy_score(yTest,textClfDTC.predict(xTest))*100

print('\nThe accuracy for the decision tree classifier is ' + str(accuracyDTC))

# k = 10 cross fold validation on decision tree classifier model.

scoresCVDTC = cross_val_score(estimator = textClfDTC, X = xTrain, y = yTrain
                             , cv = 10, n_jobs = 1)

# Storing the accuracy and standard deviation in variables.

accuracyCVDTC = np.mean(scoresCVDTC)*100

print('\nThe accuracy for k = 10 cross fold validation on decision tree',
      ' classifier model',' is ' + str(accuracyCVDTC),sep = '')

stdCVDTC = np.std(scoresCVDTC)

print('\nThe standard deviation for k = 10 cross fold validation on decision'
      ' tree classifier Model',' is ' + str(stdCVDTC),sep = '')

# Importing the "RandomForestClassifier" package for using random forest (RF) models.

from sklearn.ensemble import RandomForestClassifier

# Creating a pipeline for RF classifier.

textClfRF = Pipeline([('vect', TfidfVectorizer())
                      ,('tfidf', TfidfTransformer()),
                      ('clf', RandomForestClassifier()),]
                     )

textClfRF.fit(xTrain,yTrain)

# Storing the accuracy of the RF model in a variable.

accuracyRF = accuracy_score(yTest,textClfRF.predict(xTest))*100

print('\nThe accuracy for the RF model is ' + str(accuracyRF))

# k = 10 cross fold validation on RF model.

scoresCVRF = cross_val_score(estimator = textClfRF, X = xTrain, y = yTrain
                             , cv = 10, n_jobs = 1)

# Storing the accuracy and standard deviation in variables.

accuracyCVRF = np.mean(scoresCVRF)*100

print('\nThe accuracy for k = 10 cross fold validation on RF',
      ' classifier model',' is ' + str(accuracyCVRF),sep = '')

stdCVRF = np.std(scoresCVRF)

print('\nThe standard deviation for k = 10 cross fold validation on RF Model'
      ,' is ' + str(stdCVRF),sep = '')

# Importing the "LogisticRegression" package for using logistic regression models.

from sklearn.linear_model import LogisticRegression

# Creating a pipeline for logistic regression classifier.

textClfLR = Pipeline([('vect', TfidfVectorizer())
                      ,('tfidf', TfidfTransformer()),
                      ('clf', LogisticRegression()),]
                     )

textClfLR.fit(xTrain,yTrain)

# Storing the accuracy of the logistic regression model in a variable.

accuracyLR = accuracy_score(yTest,textClfLR.predict(xTest))*100

print('\nThe accuracy for the logistic regression model is ' + str(accuracyLR))

# k = 10 cross fold validation on logistic regression model.

scoresCVLR = cross_val_score(estimator = textClfLR, X = xTrain, y = yTrain
                             , cv = 10, n_jobs = 1)

# Storing the accuracy and standard deviation in variables.

accuracyCVLR = np.mean(scoresCVLR)*100

print('\nThe accuracy for k = 10 cross fold validation on logistic regression',
      ' classifier model',' is ' + str(accuracyCVLR),sep = '')

stdCVLR = np.std(scoresCVLR)

print('\nThe standard deviation for k = 10 cross fold validation on logistic'
      ' regression classifier Model',' is ' + str(stdCVLR),sep = '')

# Importing the "SVC" package for using Support Vector Machines (SVM) models.

from sklearn.svm import SVC

# Creating a pipeline for SVC classifier.

textClfSVC = Pipeline([('vect', TfidfVectorizer())
                      ,('tfidf', TfidfTransformer()),
                      ('clf', SVC()),]
                     )

textClfSVC.fit(xTrain,yTrain)

# Storing the accuracy of the SVC model in a variable.

accuracySVC = accuracy_score(yTest,textClfSVC.predict(xTest))*100

print('\nThe accuracy for the SVC model is ' + str(accuracySVC))

# k = 10 cross fold validation on SVC model.

scoresCVSVC = cross_val_score(estimator = textClfSVC, X = xTrain, y = yTrain
                             , cv = 10, n_jobs = 1)

# Storing the accuracy and standard deviation in variables.

accuracyCVSVC = np.mean(scoresCVSVC)*100

print('\nThe accuracy for k = 10 cross fold validation on SVC',
      ' classifier model',' is ' + str(accuracyCVSVC),sep = '')

stdCVSVC = np.std(scoresCVSVC)

print('\nThe standard deviation for k = 10 cross fold validation on SVC Model'
      ,' is ' + str(stdCVSVC),sep = '')

# Creating two lists to store the accuracies of the models and their names.

modelAccuracyList = [accuracyCVNB,accuracyCVDTC,accuracyCVRF,accuracyCVLR,accuracyCVSVC]

modelNamesList = ['Multinomial Naive Bayes', 'Decision Tree Classifier'
                  ,'Random Forest', 'Logistic Regression'
                  ,'Support Vector Machines'
                  ]

# A bar plot is created to compare the models investigated during model selection.

plt.figure()

# Generating the bar plot.

plot1 = sns.barplot(x = modelNamesList,y = modelAccuracyList)

# Rotating the x-axis labels by 30 degrees.
    
plot1.set_xticklabels(plot1.get_xticklabels(),rotation = 30)

# Adding title to the plot.

plt.title('Barplot for comparing the models from their accuracy')

# Adding the xlabel to the plot.

plot1.set_xlabel('Model Name')

# Adding the ylabel to the plot.

plot1.set_ylabel('Model Accuracy')

# Displaying the plot.

plt.show()

# The plot shows that the model with the highest accuracy is SVM classifier.

# Storing the index of the model with highest accuracy.

modelMaxAccuracyIndex = modelAccuracyList.index(max(modelAccuracyList))

# Storing the model name with the highest accuracy.

modelNameMaxAccuracy = modelNamesList[modelMaxAccuracyIndex]

# The model with the highest accuracy is SVM classifier with an accuracy of 98.93%.
# Hence, that model will be saved using Python module 'Pickle'.

import pickle

# Storing the filename for the model in a string variable.

modelFileName = filePath1 + filePath2 + 'FinalisedModel.sav'

pickle.dump(textClfSVC, open(modelFileName, 'wb'))

# Loading the saved Model.

loadedModel = pickle.load(open(modelFileName, 'rb'))

# Estimating the performance of the model using the test dataset.

loadedModel.fit(xTrain,yTrain)

# Storing the accuracy of the loaded SVC model in a variable.

accuracyLoaded = accuracy_score(yTest,loadedModel.predict(xTest))*100

print('\nThe accuracy for the loaded SVC model is ' + str(accuracyLoaded))

# k = 10 cross fold validation on loaded SVC model.

scoresCVLoaded = cross_val_score(estimator = loadedModel, X = xTest, y = yTest
                                 , cv = 10, n_jobs = 1)

# Storing the accuracy and standard deviation in variables.

accuracyCVLoaded = np.mean(scoresCVLoaded)*100

print('\nThe accuracy for k = 10 cross fold validation on loaded SVC',
      ' classifier model',' is ' + str(accuracyCVLoaded),sep = '')

stdCVLoaded = np.std(scoresCVLoaded)

print('\nThe standard deviation for k = 10 cross fold validation on loaded SVC'
      ,' Model is ' + str(stdCVLoaded),sep = '')

# Creating a list to compute the cross fold validation of SVC for different 'k'
# values.

cvList = [10,15,20,25,30,35]

# Creating a list to store the accuracy for different 'k' cross fold validation
# models.

accuracyCVList = []

# Creating a list to store the standard deviation for different 'k' cross fold 
# validation models.

stdCVList = []

for cv in cvList:
    
    # Cross fold validation on loaded SVC model for different 'k' values.
    
    scoresCVLoaded = cross_val_score(estimator = loadedModel, X = xTest
                                     , y = yTest, cv = cv, n_jobs = 1)
    
    # Storing the accuracy and standard deviation in variables.
    
    accuracyCVLoaded = np.mean(scoresCVLoaded)*100
    
    print('\nThe accuracy for k = ',cv,' cross fold validation on loaded SVC',
          ' classifier model',' is ' + str(accuracyCVLoaded),sep = '')
    
    stdCVLoaded = np.std(scoresCVLoaded)
    
    print('\nThe standard deviation for k = ',cv,' cross fold validation on' 
          'loaded SVC',' Model is ' + str(stdCVLoaded),sep = '')
    
    accuracyCVList.append(accuracyCVLoaded)
    
    stdCVList.append(stdCVLoaded)
    
# A line plot is created to compare the accuracy of the models for different 
# 'k' values.

plt.figure()

# Generating the Line plot. 

plot1 = sns.lineplot(x = cvList, y = accuracyCVList)

# Adding title to the plot.

plt.title("Lineplot of model accuracy vs 'k'")

# Adding the xlabel to the plot.

plot1.set_xlabel('k')

# Adding the ylabel to the plot.

plot1.set_ylabel('Model Accuracy')

# A red horizontal line is added to the plot to denote the overall average accuracy.
      
plot1.axhline(np.mean(accuracyCVList),color = "red")

# Displaying the plot.

plt.show()

# A line plot is created to compare the out-of-sample error of the models for 
# different 'k' values.

plt.figure()

# Generating the Line plot. 

plot1 = sns.lineplot(x = cvList, y = stdCVList)

# Adding title to the plot.

plt.title("Lineplot of model out-of-sample error vs 'k'")

# Adding the xlabel to the plot.

plot1.set_xlabel('k')

# Adding the ylabel to the plot.

plot1.set_ylabel('Out-of-sample Error')

# A yellow horizontal line is added to the plot to denote the overall average 
# out-of-sample error.
      
plot1.axhline(np.mean(stdCVList),color = "yellow")

# Displaying the plot.

plt.show()

# The best value for 'k' lies between 10 to 20 which can be observed from the
# plots. This way we can benchmark any model based on its accuracy and
# out-of-sample error.
