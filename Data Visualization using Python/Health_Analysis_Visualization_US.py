# The "pandas" package is used for processing data stored in CSV files.

import pandas as pd

# The "pandas.api.types" package contains functions that can check the 
# datatype of a variable in a pandas dataframe or Series.

import pandas.api.types as ty

# Since there would be some visualisation involved, we would be using the 
# "matplotlib.pyplot" and "seaborn" packages.

import matplotlib.pyplot as plt

import seaborn as sns

# Storing the file path of the datasets in a variable for better readability.

filePath1 = 'C:/Users/prash/Downloads/MSc Data Science and Analytics/'

filePath2 = 'Scientific Programming in Python/ProjectSpec_Dataset/'
                  
filePath = filePath1 + filePath2
            
# Storing the filenames of the datasets in a list variable.

dataSetFileNames = ['NhanesDemoAdapted.csv','NhanesFoodAdapted.csv']

# Storing the datasets in pandas dataframes.

dataFrameNhanesDemoAdapted = pd.read_csv(filePath + dataSetFileNames[0])

dataFrameNhanesFoodAdapted = pd.read_csv(filePath + dataSetFileNames[1])

# Using the describe function to familiarise with the contents of each dataframe.

dataFrameNhanesDemoAdapted.describe()

dataFrameNhanesFoodAdapted.describe()

# Before performing any operation in the datasets, the data cleaning must be
# done. First, we need to identify the data that has categorical information.

# In the Demo dataset, the variables "US born", "Education", "Marital Status", 
# "HouseholdSize" are categorical data, despite holding numeric values.
# These variables/predictors must be converted to string type.

# This does not apply to the Diet dataset, since there is no categorical
# information.

# Storing the predictor names of Demo dataset in a temporary list.

tempNamesDemo = ["US born", "Education", "Marital Status", "HouseholdSize"]

for tempNameDemo in tempNamesDemo:
    
    # Storing the converted data in a variable. If there is any missing data,
    # it would also be converted as string. For example, nan or na would 
    # become "nan" or "na" respectively.
    
    demoNum2Str = dataFrameNhanesDemoAdapted[tempNameDemo].astype(str)
    
    # Replacing the numeric data with string data in the dataset.
    
    dataFrameNhanesDemoAdapted[tempNameDemo] = demoNum2Str
                           
# If there are missing values in the numeric/float data in a column, then
# they can be replaced with its mean. 

# If there are missing values in the categorical data in a column, then
# they can be replaced with its mode. 

# Replacing missing values in the Demo dataset.

# Storing the column names of Demo dataset in a variable.
 
columnNamesDemo = dataFrameNhanesDemoAdapted[:0]

for columnNameDemo in columnNamesDemo:
    
    # Checking if the column contains integer or float values.
    
    boolInt = ty.is_integer_dtype(dataFrameNhanesDemoAdapted[columnNameDemo])
    
    boolFloat = ty.is_float_dtype(dataFrameNhanesDemoAdapted[columnNameDemo])
    
    if boolInt or boolFloat:
        
        # Storing the column information in a temporary Pandas Series.

        tempDf = dataFrameNhanesDemoAdapted[columnNameDemo]
        
        # Checking rows having missing values in the temporary Series.
        
        if sum(tempDf.isna()) != 0:
            
            # Storing the mean in a variable.
            
            tempAvg = tempDf.mean()
            
            # Replacing the missing data in the dataset with its mean.
            
            dataFrameNhanesDemoAdapted[columnNameDemo] = tempDf.fillna(tempAvg)
            
    # Checking if the column contains other data types such as string.
        
    else:

        # Storing the column information in a temporary Pandas Series.
        
        dfCategorical = dataFrameNhanesDemoAdapted[columnNameDemo]
        
        # Checking rows having missing values in the temporary Series.
        
        if sum(dfCategorical == 'nan') != 0:
            
            # Storing the series data without 'nan' in a variable. 
            
            dfWithoutNa = dfCategorical[dfCategorical != 'nan']
            
            # Storing the count per category in a variable. 
            
            dfCatCount = dfWithoutNa.value_counts()
            
            # Storing the mode in a variable.
            
            tempMode = dfCatCount[dfCatCount == dfCatCount.max()].index
            
            tempMode = str(tempMode.values[0])
                        
            # Replacing the missing data in the dataset with its mode.
              
            dataFrameNhanesDemoAdapted.loc[dfCategorical == 'nan'
                                           ,columnNameDemo] = tempMode
            
# Replacing missing values in the Diet dataset.

# Storing the column names of Diet dataset in a variable.
 
columnNamesDiet = dataFrameNhanesFoodAdapted[:0]

for columnNameDiet in columnNamesDiet:
    
    # Checking if the column contains integer or float values.
    
    boolInt = ty.is_integer_dtype(dataFrameNhanesFoodAdapted[columnNameDiet])
    
    boolFloat = ty.is_float_dtype(dataFrameNhanesFoodAdapted[columnNameDiet])
    
    if boolInt or boolFloat:
        
        # Storing the column information in a temporary Pandas Series.

        tempDf = dataFrameNhanesFoodAdapted[columnNameDiet]
        
        # Checking rows having missing values in the temporary Series.
        
        if sum(tempDf.isna()) != 0:
            
            # Storing the mean in a variable.
            
            tempAvg = tempDf.mean()
            
            # Replacing the missing data in the dataset with its mean.
            
            dataFrameNhanesFoodAdapted[columnNameDiet] = tempDf.fillna(tempAvg)
            
    # Checking if the column contains other data types such as string.
    
    else:

        # Storing the column information in a temporary Pandas Series.
        
        dfCategorical = dataFrameNhanesFoodAdapted[columnNameDiet]
        
        # Checking rows having missing values in the temporary Series.
        
        if sum(dfCategorical == 'nan') != 0:
            
            # Storing the series data without 'nan' in a variable. 
            
            dfWithoutNa = dfCategorical[dfCategorical != 'nan']
            
            # Storing the count per category in a variable. 
            
            dfCatCount = dfWithoutNa.value_counts()
            
            # Storing the mode in a variable.
            
            tempMode = dfCatCount[dfCatCount == dfCatCount.max()].index
            
            tempMode = str(tempMode.values[0])
            
            # Replacing the missing data in the dataset with its mode.
            
            dataFrameNhanesFoodAdapted.loc[dfCategorical == 'nan'
                                           ,columnNameDemo] = tempMode
            
# Apparently, there are no missing values in the Diet Dataset.            
            
# Creation of function for displaying a Menu.

# This function will take two dataframes as input, the first containing the 
# demographic data, the second containing the dietary data.
# The food dataset will be reduced (using pandas groupby functionality) to
# hold the average food intake per meal per individual. 
# This new dataframe (containing only one row per individual) will then be 
# merged with the demographic dataframe via pandas, such that it
# only contains entries for individuals that were in the Food dataset.
# The code should then write this merged dataframe to a csv file called
# (‘studentName_merged.csv’). 

def mergeDatasets(dfDemo,dfFood):
    
    # Storing the reduced food data in a variable.
    
    dfFoodReduced = dfFood.groupby('SEQN',as_index = False).mean()
    
    # Merging the reduced food data with demo data using inner join.
    
    dfMerged = pd.merge(dfDemo,dfFoodReduced,on = 'SEQN',how = 'inner')
    
    # Creating a copy of the nerged dataframe.
    
    dfMergedCopy = dfMerged
    
    # Writing the merged data to a csv file called ‘studentName_merged.csv’
    
    dfMerged.to_csv(filePath + 'studentName_merged.csv',index = False)
    
    # Returning the merged dataset.
    
    return dfMergedCopy
    
# Creation of menu.

# The outer while loop will run infinite times, until the option 5 (for exit) 
# is provided.

while 1 :
    
    # The inner while loop (menu window) will run infinite times, until the 
    # proper input option is provided. If the entered input is not an integer,
    # then a message will be displayed to the user to enter an integer input.
    # If the entered input is not from the mentioned options, then a message
    # will be displayed to the user for entering an appropriate input choice.

    while 1:
        
        try:
            
            # Storing the input prompt in separate variables for readability.
            
            inputStr1 = 'Please select one of the following options:\n'
            
            inputStr2 = '\n1. Household income per ethnicity'
            
            inputStr3 = '\n2. Marital status\n3. Income and education level'
            
            inputStr4 = '\n4. Diet analysis\n5. Exit\n\n'
            
            inputPrompt = inputStr1 + inputStr2 + inputStr3 + inputStr4
            
            menuInput = int(input(prompt = inputPrompt))
            
            if menuInput > 0 and menuInput < 6:
                break        
            
            print("\nInvalid input. Please re-enter the input once again.")
            
        except:
            
            print("\nPlease enter an integer input.")    
    
    print('\n')
    
    # Menu Option 1 – Household Income of ethnicities :-
    
    # In order to assess if there are societal disadvantages we 
    # investigate the relationship between the average HouseholdIncome attained 
    # by respondents and their ethnicity. The user should first be told how 
    # many ethnicities are represented, and a sorted list of number of 
    # respondents per ethnicity. The average HouseholdIncome per ethnicity 
    # (only considering adult respondents) should then be conveyed using a 
    # horizontal bar graph. 
    
    if menuInput == 1:
       
      # For displaying the total number of ethnicities and the 
      # sorted list of number of respondents per ethnicity, we need to store 
      # the 'Ethnicity' predictor in a separate dataframe variable.
      
      # The 'Ethnicity' predictor is stored in a dataframe variable.
      
      dataFrameNhanesEthnicity = dataFrameNhanesDemoAdapted['Ethnicity']
      
      # Storing the count of unique Ethnicities from the dataset in a variable.
      
      ethnicityCount = len(pd.unique(dataFrameNhanesEthnicity))
      
      # Displaying the Number of Ethnicities in the dataset.
      
      print('Number of Ethnicities in the dataset:',ethnicityCount)
         
      # Displaying the sorted list of number of respondents per ethnicity.
      
      print('Number of respondents per Ethnicity:\n'
            ,dataFrameNhanesEthnicity.value_counts(),'\n'
            ,sep = ''
            )
      
      # Storing the data of Adult respondents (aged 20 and over) in a variable.
      
      boolAdults = dataFrameNhanesDemoAdapted['Age'] > 19
      
      dfNhanesAdults = dataFrameNhanesDemoAdapted[boolAdults]
      
      # Storing the average household incomes of adults in a variable.
      
      avgHouseholdIncome = round(dfNhanesAdults['HouseholdIncome'].mean(),3)
      
      # Storing the data of adults grouped by their ethnicity in a variable.
      
      ethinicityGroup = dfNhanesAdults.groupby('Ethnicity')
      
      # Storing the average HouseholdIncome per ethnicity in a variable.
      
      ethinicityAvgIncomeGroup = ethinicityGroup['HouseholdIncome'].mean()
      
      print('\nThe average household income of each adults =\n\n'
            ,ethinicityAvgIncomeGroup,sep = ''
            )
      
      print('\nThe average household income of all the adults ='
            ,avgHouseholdIncome
            )
        
      # Plotting the horizontal bar graph to convey the average 
      # HouseholdIncome per ethnicity.
      
      # Creating a new plot.
      
      plt.figure()
      
      # Storing the plot title as string in a variable.
      
      plotTitle = 'Horizontal Bar plot of Ethnicity vs Average Household Income\n' 
      
      # Generating the bar plot.
      
      plot1 = ethinicityAvgIncomeGroup.plot(kind = 'barh', title = plotTitle)
      
      # Adding the xlabel to the plot.
      
      plot1.set_xlabel('Average Household Income')
      
      # A red vertical line is added to the plot to denote the average income
      # of all the adults.
      
      plot1.axvline(avgHouseholdIncome,color = "red")
      
      # Adding the legend to the plot.
      
      plot1.legend(['Avg Income (All)','Average Income'])
      
      # Displaying the plot.
      
      plt.show()
      
      # Among the adults, the Asians hold the highest average household income
      # and the Mexican-Americans hold the least average household income.
      # The average income of Blacks is higher than the Hispanics'. 
      # The Others have the second least average income. When compared to the
      # average income of all the adults, the average of the Asians is higher.
      # The rest of the averages fall below the overall average. This means
      # that there are social disadvantages as the Asians are the only people
      # with the highest average income, while the rest are below average.
    
    # Menu Option 2 – Marital status :-
    
    # Investigate the relationship between age and marriage for adult 
    # respondents only (aged 20 and over). First print out a sorted 
    # list indicating the number of respondents per marital status category.
    # Using a line graph it should display the 1st, 2nd, and 3rd quartiles of 
    # age for each status type. The line graph should have three lines, 
    # one for each quartile. (Note the 1st quartile is also referred to as the 
    # 25th percentile, the 2nd as the median and the 3rd as the 75th percentile.)
    
    elif menuInput == 2:
          
      # For displaying the sorted list of number of adult respondents 
      # per marital status category, we need to store the 'Marital Status' 
      # predictor from the Adult respondents' dataset in a variable.
      
      # Storing the data of Adult respondents (aged 20 and over) in a variable.
      
      boolAdults = dataFrameNhanesDemoAdapted['Age'] > 19
      
      dfNhanesAdults = dataFrameNhanesDemoAdapted[boolAdults]
      
      dfAdultMaritalStats = dfNhanesAdults['Marital Status']
    
      # Displaying the sorted list of number of respondents per marital status
      # category.
      
      print('Number of respondents per marital status category:\n'
            ,dfAdultMaritalStats.value_counts(),'\n'
            ,sep = ''
            )
      
      # Storing the data of adults grouped by their marital status in a variable.
      
      maritalStatsGroup = dfNhanesAdults.groupby('Marital Status')
      
      # Storing the summary of Age per Marital Status category in a variable.
      
      maritalStatsAgeGroup = maritalStatsGroup['Age'].describe()
      
      print('The summary of each marital status category:\n\n'
            ,maritalStatsAgeGroup,sep = ''
            )
      
      # Storing the marital status indexes in a variable.
      
      maritalStatsIndexes = maritalStatsAgeGroup.index.values
      
      # Plotting the line graph of age vs the quartiles for each status type.
      
      # Iterating through each status type.
      
      for maritalStatsIndex in maritalStatsIndexes:
          
          # Storing the marital status in a variable based on the indexes
          # mentioned below. This is taken from the problem statement.
          
          # {1:Married, 2:Widowed, 3:Divorced, 4:Separate, 5:Single
          # , 6: Living with Partner}
          
          statName = 'Living with Partner' 
          
          if maritalStatsIndex == '1.0':
              statName = 'Married'
                 
          elif maritalStatsIndex == '2.0':
              statName = 'Widowed'
             
          elif maritalStatsIndex == '3.0':
              statName = 'Divorced'
                 
          elif maritalStatsIndex == '4.0':
              statName = 'Separate'
          
          elif maritalStatsIndex == '5.0':
              statName = 'Single'
          
          # Storing the age of a respondent for each status type in a variable.
          
          boolAgeStats = dfNhanesAdults['Marital Status'] == maritalStatsIndex
          
          dfAgeMarStat = dfNhanesAdults['Age'][boolAgeStats]
          
          # Storing the quartiles of the age in separate variables.
          
          ageQ1 = maritalStatsAgeGroup['25%'][maritalStatsIndex]
          
          ageQ2 = maritalStatsAgeGroup['50%'][maritalStatsIndex]
          
          ageQ3 = maritalStatsAgeGroup['75%'][maritalStatsIndex]
          
          # Plotting the line graph of Age for each marital status.
          
          # Creating a new plot.
          
          plt.figure()
          
          # Storing the plot title as string in a variable.
          
          pl_title = 'Line graph of Age for marital status "' + statName + '"'
          
          # Generating the line plot.
          
          plt.plot(dfAgeMarStat.values,color = 'black')
          
          # Adding the xlabel to the plot.
          
          plt.ylabel('Age')
          
          # Adding the xlabel to the plot.
          
          plt.xlabel('Index')
          
          # Adding the title to the plot.
          
          plt.title(pl_title)
                                  
          # A blue horizontal line is added to the plot to denote the 25th 
          # percentile of age.
          
          plt.plot(range(0,len(dfAgeMarStat))
                   ,[ageQ1]*len(dfAgeMarStat)
                   ,color = "blue"
                   )
          
          # A red horizontal line is added to the plot to denote the 50th 
          # percentile (median) of age.
          
          plt.plot(range(0,len(dfAgeMarStat))
                   ,[ageQ2]*len(dfAgeMarStat)
                   ,color = "red"
                   )
          
          # A green horizontal line is added to the plot to denote the 75th 
          # percentile of age.
          
          plt.plot(range(0,len(dfAgeMarStat))
                   ,[ageQ3]*len(dfAgeMarStat)
                   ,color = "green"
                   )
          
          # Saving the quartiles for the legend in a list variable.
          
          quartList = ['25th percentile','50th percentile','75th percentile']
          
          # Adding the legend to the plot.
          
          plt.legend(['Age'] + quartList)
          
          # Displaying the plot.
      
          plt.show()
          
      # Out of all the adults, the highest percentage belongs to the 
      # Married people. Among them, 25% of peoples' age is lesser than 41,
      # 50% of peoples' age is less than 55 and 75% of peoples' age is more
      # than or equal to 66. 
      
      # The next highest percentage is from the category of singles.
      # Among them, 25% of peoples' age is lesser than 24, 50% of peoples' age 
      # is less than 31 and 75% of peoples' age is more than or equal to 46.
      
      # The 3rd highest percentage is from the category of divorced people.
      # Among them, 25% of peoples' age is lesser than 51, 50% of peoples' age 
      # is less than 60 and 75% of peoples' age is more than or equal to 68.
      
      # The 4th highest percentage is from the category of people living with
      # their partners.
      # Among them, 25% of peoples' age is lesser than 28, 50% of peoples' age 
      # is less than 37 and 75% of peoples' age is more than or equal to 50.
      
      # The 2nd least percentage is from the category of people who are 
      # widowed.
      # Among them, 25% of peoples' age is lesser than 65, 50% of peoples' age 
      # is less than 75 and 75% of peoples' age is more than or equal to 80.
      
      # The least percentage is from the category of people who are separated
      # from their partners.
      # Among them, 25% of peoples' age is lesser than 40, 50% of peoples' age 
      # is less than 52 and 75% of peoples' age is more than or equal to 63.
      
    # Menu Option 3 – Education and income :-
    
    # In this case, we wish to assess the correlation between education 
    # and income. The user will be given the option of
    # 1. IncomePovertyRatio
    # 2. HouseholdIncome
    
    # A simple bar graph comparing the average of the user option for the 
    # different education levels will be given. For example, if the user selects
    # the second option then a bar graph containing the average income for each 
    # education level should be generated. Your report should contain a sample 
    # bar graph that is outputted and discussion of the results depicted in 
    # the graph (e.g. does higher level of education suggest a higher income?).
    
    elif menuInput == 3:
          
      # A new list will be created to store the column names mentioned earlier.
      
      optList = ['IncomePovertyRatio','HouseholdIncome']
      
      # The user has to choose from this list's index. If the entered input is
      # not an integer, then a message will be displayed to the user to enter 
      # an integer input. If the entered input is not from the mentioned 
      # options, then a message will be displayed to the user for entering an 
      # appropriate input choice.
      
      # This can be implemented using an infinite while loop.
      
      while 1:
          
          try:
              
              # Generating an input prompt from the option list.
              
              inputStr1 = 'Please select one of the following options:\n\n'
              
              inputStr3 = ''
           
              for optionIndex in range(0,len(optList)):
                  
                  inputStr2 = str(optionIndex + 1) + '. '
                  
                  inputStr3 = inputStr3 + inputStr2 + optList[optionIndex] + '\n'
                  
              inputPrompt = inputStr1 + inputStr3 + '\n'
          
              menuInput = int(input(prompt = inputPrompt))
          
              if menuInput > 0 and menuInput < (len(optList) + 1):
                  break
              
              print("\nInvalid input. Please re-enter the input once again.")
          
          except:            
              
            print("\nPlease enter an integer input.")  
      
      # Storing the selected column in a variable.
      
      colSelected = optList[menuInput - 1]
            
      # Storing the average income of all respondents in a variable.
      
      avgIncome = round(dataFrameNhanesDemoAdapted[colSelected].mean(),3)
      
      # Grouping the data by Education predictor and storing it in a variable.
      
      educationGroup = dataFrameNhanesDemoAdapted.groupby('Education')
      
      # Storing the average Income per education in a variable.
      
      educationAvgIncomeGroup = educationGroup[colSelected].mean()
      
      print('\nThe average income of each education category =\n\n'
            ,educationAvgIncomeGroup,sep = ''
            )
      
      print('\nThe average income of all the respondents ='
            ,avgIncome
            )
        
      # Plotting the bar graph to convey the average Income per education.
      
      # Creating a new plot.
      
      plt.figure()
      
      # Storing the plot title as string in a variable.
      
      plotTitle = 'Bar plot of ' + colSelected + ' vs Education Category\n' 
      
      # Generating the bar plot.
      
      plot1 = educationAvgIncomeGroup.plot(kind = 'bar', title = plotTitle)
      
      # Adding the xlabel to the plot.
      
      plot1.set_xlabel('Education Category')
      
      # Adding the ylabel to the plot.
      
      plot1.set_ylabel(colSelected)
      
      # A red vertical line is added to the plot to denote the average income
      # of all the respondents.
      
      plot1.axhline(avgIncome,color = "red")
      
      # Adding the legend to the plot.
      
      plot1.legend(['Avg Income (All)','Average Income'],loc = 'best')
      
      # Displaying the plot.
      
      plt.show()
    
    # {1:<9th Grade, 2: 9th-11th grade, 3:HighSchool graduate, 4:Some college
    #  , 5: College graduate or above}
    
    # The IncomePovertyRatio is 99% uniform across all education categories.
    # The average IncomePovertyRatio of all respondents = 2.375.
    # Category 0 has the highest ratio, whereas the people with 9th-11th grade
    # as the highest education level hold the 2nd highest average income ratio. 
    # The income of respondents who at max have completed 9th grade have an 
    # income greater than people who have graduated in some college. 
    # Respondents who are either High School graduates or college graduates 
    # hold the similar income.
    # Overall, 67% of the categories fall slightly below the average line and 
    # 33% of the categories fall slightly above the average line. 
    # This is almost a uniform distribution.
    
    # The HouseholdIncome is almost uniform across all education categories.
    # The average HouseholdIncome of all respondents = 139.476.
    # Category 0 has the least income, whereas the people with 9th-11th grade
    # as the highest education level hold the highest average income. 
    # The income of respondents who at max have completed 9th grade have an 
    # income greater than people who have graduated in some college. Both
    # categories are just above the overall average mark. Respondents who are
    # either High School graduates or college graduates hold the similar income.
    # Overall, 50% of the categories fall slightly below the average line and 
    # 50% of the categories fall slightly above the average line. This is akin 
    # to a uniform distribution.
    
    # Higher level of education does not mean that one can get a higher income.
    
    # Menu Option 4 – Diet :-
    
    # The function should take as input two dataframes, the first containing 
    # the demographic data, the second containing the dietary data.
    # First, the food dataset (using pandas groupby functionality) must be reduced
    # to the average food intake per meal per individual. This new dataframe 
    # (containing only one row per individual) will then be merged with the 
    # demographic dataframe via pandas, such that it only contains entries for 
    # individuals that were in the Food dataset. The code should then write 
    # this merged dataframe to a csv file called (‘studentName_merged.csv’). 
    
    # For this merged dataframe, the user will be given the list of nutrients and 
    # asked to select a category. Boxplots will then be generated for the categories of Gender, 
    # Ethnicity and Education. Scatter plot will be generated for comparing with 
    # HouseholdIncome, and comparing with Age.
    
    elif menuInput == 4:
          
      # The function for merging the datasets 'mergeDatasets' 
      # is defined outside the while loops.
      
      # Running the 'mergeDatasets' function and storing the returned merged
      # dataset in a dataframe.
      
      dfMerged = mergeDatasets(dataFrameNhanesDemoAdapted
                               ,dataFrameNhanesFoodAdapted
                               )
      
      # A new list will be created to store the nutrients' information which 
      # is nothing but the column names of food dataset excluding the Sequence 
      # field 'SEQN'.
      
      optList = list(columnNamesDiet)[1:]
      
      # The user has to choose from this list's index. If the entered input is
      # not an integer, then a message will be displayed to the user to enter 
      # an integer input. If the entered input is not from the mentioned 
      # options, then a message will be displayed to the user for entering an 
      # appropriate input choice.
      
      # This can be implemented using an infinite while loop.
      
      while 1:
          
          try:
              
              # Generating an input prompt from the option list.
              
              inputStr1 = 'The following nutrients are available\n\n'
              
              inputStr3 = ''
              
              inputStr4 = 'Which category do you wish (please enter the number)'
              
              for optionIndex in range(0,len(optList)):
                  
                  inputStr2 = str(optionIndex + 1) + ' '
                  
                  inputStr3 = inputStr3 + inputStr2 + optList[optionIndex] + '\n'
                  
              inputPrompt = inputStr1 + inputStr3 + '\n' + inputStr4 + '\n\n'
          
              menuInput = int(input(prompt = inputPrompt))
          
              if menuInput > 0 and menuInput < (len(optList) + 1):
                  break
              
              print("\nInvalid input. Please re-enter the input once again.")
          
          except:            
              
            print("\nPlease enter an integer input.")  
      
      # Storing the selected column in a variable.
      
      colSelected = optList[menuInput - 1]
      
      # Generating the boxplots for the categories of Gender, Ethnicity 
      # and Education.
      
      # Storing the column names for the boxplot variables in a list.
      
      boxPlotList = ['Gender','Ethnicity','Education']
      
      # Generating boxplots via loop.
      
      for boxPlotCol in boxPlotList:
              
          # Creating a new plot.
          
          plt.figure()
          
          # Storing the plot title as string in a variable.
          
          pl_title = 'Boxplot of ' + colSelected + ' vs ' + boxPlotCol + '\n' 
          
          # Generating the boxplot.
          
          ax = sns.boxplot(x = dfMerged[boxPlotCol].values
                           ,y = dfMerged[colSelected].values
                           )
          
          # Rotating the x-axis labels by 30 degrees.
          
          ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
          
          # Adding the title to the plot.
          
          plt.title(pl_title)

          # Adding the ylabel to the plot.
          
          plt.ylabel(colSelected)
          
          # Adding the xlabel to the plot.
          
          plt.xlabel(boxPlotCol)
          
          # Displaying the plot.
      
          plt.show()
        
      # Generating the Scatterplots for comparing with HouseholdIncome, 
      # and comparing with Age.
      
      # Storing the column names for the Scatterplot variables in a list.
      
      scatterPlotList = ['HouseholdIncome','Age']
      
      # Generating scatterplots via loop.

      for scatterPlotCol in scatterPlotList:
              
          # Creating a new plot.
          
          plt.figure()
          
          # Storing the plot title as string in a variable.
          
          pl_title = 'Scatterplot of ' + colSelected + ' vs ' + scatterPlotCol + '\n' 
          
          # Generating the Scatter Plot.
          
          sns.scatterplot(x = dfMerged[scatterPlotCol].values
                          ,y = dfMerged[colSelected].values
                          )
          
          # Adding the title to the plot.
          
          plt.title(pl_title)

          # Adding the ylabel to the plot.
          
          plt.ylabel(colSelected)
          
          # Adding the xlabel to the plot.
          
          plt.xlabel(scatterPlotCol)    
          
          # Displaying the plot.
      
          plt.show()
              
    # Menu Option 5 – Exit :- 
    
    # If the user selects an option 1-5 then the program should display the 
    # associated output and will subsequently display the main menu again.
    # If the user selects option 5 the application should exit. 
    # Hence, a message will be displayed before termination of the application.
    
    else:
        
        print("Thanks for using our application. Exiting now!!!\n")
        
        break
