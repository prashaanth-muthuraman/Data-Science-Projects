# Download the dataset from the following website. 
# https://archive.ics.uci.edu/ml/machine-learningdatabases/statlog/heart/

# The read.delim() function reads a file in table format.
# The data is space delimited and does not contain any headers, so it needs to 
# be added later on.

heart_df <- read.delim('heart.dat',sep=' ',stringsAsFactors = 1,header = 0)

# Assignment of headers/column names to the dataset.

colnames(heart_df) <- c("age","sex","chest_pain_type","resting_blood_pressure"
                        ,"serum_cholestoral","fasting_blood_sugar"
                        ,"resting_ecg_results","max_heart_rate_achieved"
                        ,"exercise_induced_angina","oldpeak"
                        ,"slope_peak_exercise_ST_segment"
                        ,"number_of_major_vessels"
                        ,"thal","heart_disease_prediction"
                        )

# "heart_disease_prediction" is the response variable.

response_column <- colnames(heart_df)[length(colnames(heart_df))]

# Exploratory Data Analysis (EDA): 
# It is given that the below columns are either ordered, binary or nominal. 
# Hence they need to be factorized/categorized.

# Ordered: "slope_peak_exercise_ST_segment"
# Binary: "sex", "fasting_blood_sugar", "exercise_induced_angina"
# Nominal: "resting_ecg_results", "chest_pain_type","number_of_major_vessels","thal",
# "heart_disease_prediction"

# This is done by iterating through the data set via the respective 
# column indexes(given).

for (col_name in colnames(heart_df)[c(11,2,6,9,7,3,12,13,14)]){
  heart_df[,col_name] = as.factor(heart_df[,col_name])
}

# 1. Analysis of numerical data
# -----------------------------
# A histogram can be plotted for the numerical data in the data set.
# The column names, mean and median can also be calculated and stored in separate vectors.
# This is done by iterating through the data set by the column names and 
# checking for the numerical data type.

# Creation of vectors for storing the column names, mean and median.

mean_numeric_columns <- c()
median_numeric_columns <- c()
numeric_columns <- c()

# Iterating through the dataset via column names.

for (col_name in colnames(heart_df)){
  # Checking if the data across that column is numeric
  
  if (is.numeric(heart_df[,col_name])){
    numeric_columns <- c(numeric_columns,col_name)
    mean_numeric_columns <- c(mean_numeric_columns,mean(heart_df[,col_name]))
    median_numeric_columns <- c(median_numeric_columns,median(heart_df[,col_name]))
    
    # Checking if the mean is greater than the median
    
    s = paste('For the "',col_name,'" column, mean ('
              ,round(mean(heart_df[,col_name]),2)
              ,') is greater than the median (',median(heart_df[,col_name]),')'
              ,sep = ''
              )
    
    if(mean(heart_df[,col_name]) < median(heart_df[,col_name])){
      s = paste('For the "',col_name,'" column, median ('
                ,median(heart_df[,col_name])
                ,') is greater than the mean ('
                ,round(mean(heart_df[,col_name]),2)
                ,')',sep = ''
                )
    }
    print(s,quote = FALSE)
    
    # Visualization of the numerical data using histogram
    
    hist(x = heart_df[,col_name],xlab = col_name
         ,main = paste('Histogram of ',col_name,sep = '')
         )
    
    abline(v = c(mean(heart_df[,col_name]),median(heart_df[,col_name]))
           ,col = c('Red','Blue')
           )
    
    legend("topright", legend = c('Mean','Median'),
           col=c("red", "blue"), lty=1:1, cex=0.8,
           title="Line types", text.font=4, bg='lightblue'
           )
  }
}

# Assigning the names of the mean and median vectors for the numerical data 
# with the column names.

names(mean_numeric_columns) <- numeric_columns
names(median_numeric_columns) <- numeric_columns

# The plots/histograms of the variables "age", "resting_blood_pressure",
# "serum_cholestoral", "max_heart_rate_achieved" are like to that of 
# a normal distribution curve.
# However, the histogram of the variable "oldpeak" is like to that of a 
# negative/decreasing distribution. 

# For the "age" column, median (55) is greater than the mean (54.43)
# For the "resting_blood_pressure" column, mean (131.34) is greater than the median (130)
# For the "serum_cholestoral" column, mean (249.66) is greater than the median (245)
# For the "max_heart_rate_achieved" column, median (153.5) is greater than the mean (149.68)
# For the "oldpeak" column, mean (1.05) is greater than the median (0.8)

# 2. Analysis of categorical data
# -------------------------------

# A bar plot can be plotted for the categorical data in the data set.
# This is done by iterating through the data set by the column names and checking 
# for the factor data type.

# Iterating through the dataset via column names.

for (col_name in colnames(heart_df)){
  # Checking if the data across that column is categorical(factors)
  
  if (is.factor(heart_df[,col_name])){
    
    # Visualization of the categorical data using bar plot
    
    barplot(table(heart_df[,col_name]),xlab = col_name
            ,main = paste('Barplot of ',col_name,sep = '')
            )
  }
}

# Observations from all the bar plots:-
# -------------------------------------

# 1. There are more people in Sex category 1 than category 0.
# 2. As the type of chest pain increases, the count also increases.
# 3. Very less people have fasting blood sugar > 120 mg/dl.
# 4. Similar number of people have ecg results 0 and 2. Very rarely the result shows 1.
# 5. Very less people have exercise induced angina.
# 6. As the the slope of the peak exercise ST segment decreases, the count also decreases.
# 7. The same can be said about the number of major vessels.
# 8. Similar number of people have "normal" and "reversable defect" thal. 
# Very less people display "fixed defect" thal.
# 9. The absence of heart disease is predicted most of the times.

# Setting a seed for replication.

set.seed(750)

# Split of data -> 80% training data means 0.8*number of rows in heart data set.

heart_df_train_rows <- sample(1:nrow(heart_df), 0.8*nrow(heart_df))
heart_df_train <- heart_df[heart_df_train_rows,]

# The training data can be removed from the main data set to get the testing data.

heart_df_test <- heart_df[-heart_df_train_rows,]

# Analysing which predictor variable(s) should be used for the root node split
# Using only the categorical predictor variables for decision tree algorithm.
# Here the number of splits at the root node can equal the number of categories 
# in the predictor variable being split. 

# It is better to create a training data set that holds the categorical 
# data only. This can be done using lapply() and unlist() functions.
# lappyl() takes two arguments as input: Input vector or list and 
# function to be applied on that input.
# It then returns a list with the same length as input data, where each element  
# is the result of applying the input function to the input data.
# unlist() produces a vector from the input list.

# The training data and "is.factor" function will be passed to lapply, so that
# the resulting list will contain only the logical values. 
# TRUE denotes column is a factor, while FALSE denotes it is not.
# Using unlist(), this list can be converted into a logical indexed vector 
# which can then be used for fetching the factor columns in the training data.

heart_df_train_categorical <- heart_df_train[,unlist(lapply(heart_df_train,is.factor))]

# With this data set, entropy and information gain needs to be calculated for 
# categorical type predictors.

# First, the entropy of the response variable needs to be calculated. 
# This can be done using the proportion table, as it provides the 
# probability information that is used in calculating entropy.
# "heart_disease_prediction" is the response variable and the last column in the
# data frame.
# So, a frequency table for the response variable is created, which is then used
# for creating the proportion table.
# For any given data frame 'x', x[,length(colnames(x))] returns the last column 
# information in 'x'. Here 'x' is training data set that holds factor columns.

column_names_training <- colnames(heart_df_train_categorical)

table_response <- heart_df_train_categorical[,response_column]

prop_table_response <- prop.table(table(table_response))

# The entropy is calculated from the proportion table using Shannon's formula. 

response_entropy <- sum(-prop_table_response * log2(x = prop_table_response))

# Now the remaining entropy of each predictor w.r.t response variable needs to 
# be calculated. Information gain can then be calculated by subtracting 
# remaining entropy from response/overall entropy.
# A column name vector is to be created, which holds only the predictor column names.

# Storing the column names of the training data with categorical fields in a vector.

col_name_train_cat <- colnames(heart_df_train_categorical)

# Creating a column name vector, which holds only the predictor column names.

column_name_vec <- colnames(heart_df_train_categorical[,-length(col_name_train_cat)])

# Column entropy and information gain vectors are created for storing the 
# remaining entropy and information gain respectively for each predictor. 

remaining_entropy_vec <- c()
info_gain_vec <- c()

# Calculation of remaining entropy and information gain.

for (column_name in column_name_vec){
  
  # Creating frequency table of response and predictor.
  
  freq_table <- table(heart_df_train_categorical[,column_name]
                      ,heart_df_train_categorical[,response_column])
  
  # Creating proportion table using frequency table.
  
  prop_freq_table <- prop.table(freq_table,margin = 1)
  
  # Creating the log-probability table. 
  
  log_table <- -1*prop_freq_table*log2(x = prop_freq_table)
  
  # If probability is zero, then log(probability) returns NA. 
  # That needs to be changed to 0, since multiplication with 0 is 0.
  
  for (col in colnames(log_table)){
    log_table[,col][is.nan(log_table[,col])] = 0
  }
  
  # The sum of probabilities needs to be calculated for each category in the 
  # predictor. This gives the overall proportion table of predictor vs response.
  
  sum_log_table <- rowSums(log_table)
  
  # Creating the feature proportion table from frequency table of response and predictor.
  
  feature_prop_table <- prop.table(rowSums(freq_table))
  
  # Calculation of remaining entropy.
  
  remaining_entropy <- sum(sum_log_table*feature_prop_table)
  
  # Calculation of information gain.
  
  Information_gain <- response_entropy-remaining_entropy
  
  # Storing the remaining entropy and information gain in their respective
  # vectors. 
  
  remaining_entropy_vec <- c(remaining_entropy_vec,remaining_entropy)
  
  info_gain_vec <- c(info_gain_vec,Information_gain)
}

# Assigning the names of the Column entropy and information gain vectors 
# with the entropy column name vector.

names(remaining_entropy_vec) <- column_name_vec
names(info_gain_vec) <- column_name_vec

# Printing the remaining entropy and information gain of all predictors.

for (column_index in 1:length(column_name_vec)){
  cat('The remaining entropy of predictor "'
      ,column_name_vec[column_index],'" is '
      ,remaining_entropy_vec[column_index],'.\n'
      ,sep = ''
      )
}

for (column_index in 1:length(column_name_vec)){
  cat('The information gain of predictor "'
      ,column_name_vec[column_index],'" is '
      ,info_gain_vec[column_index],'.\n'
      ,sep = ''
      )
}

# Printing the predictor that has the minimum remaining entropy, which also yields maximum
# information gain.

print(c(min(remaining_entropy_vec),names(which.min(remaining_entropy_vec))))
print(c(max(info_gain_vec),names(which.max(info_gain_vec))))

# The "chest_pain_type" predictor has the highest information gain when compared
# to the other predictors. This is because it has the least remaining entropy.

# Hence, the "chest_pain_type" predictor is to be used for the root node split.
# If other predictors are used, then the tree may become complex and more diverse.
# It will be hard to predict with this model, as there would be more errors
# and high bias. There is high chance for misclassification.

root_node_name <- names(which.max(info_gain_vec))
factor(levels(heart_df_train_categorical[,root_node_name]))

# The "chest_pain_type" predictor contains only 4 values. So, the root node split
# will occur based on these values. Thus, the number of splits is equal to 4.

table(heart_df_train_categorical[,root_node_name])
prop.table(table(heart_df_train_categorical[,root_node_name]))

# The highest percentage of people suffer from chest pain Type "4", while the 
# least suffer from chest pain Type "1". Hence it would be easier to predict
# if people are suffering from chest pain Type "4" than chest pain Type "1".
# However, people suffering from Type "1" have the lowest number of observations.
# This means the tree becomes less complex and diverse as it goes down.

# Redoing above steps but this time using binary split, i.e. a split with only 
# 2 possible outcomes. 

# Since the "chest_pain_type" predictor contains 4 values,
# a binary split must be done with the value that has the highest percentage or
# probability. One outcome will hold "Yes" if type matches the one with 
# max probability. Else it will be "No". A new column in the training data
# can be added to store this in a categorical fashion.

tbl_root_node_type <- table(heart_df_train_categorical[,root_node_name])

prop_tbl_root_node_type <- prop.table(tbl_root_node_type)

max_name <- names(which.max(prop_tbl_root_node_type))

node_type <- as.factor(ifelse(heart_df_train_categorical[,root_node_name] == max_name
                              ,"Yes","No"
                              )
                       )

heart_df_train_categorical$node_type_binary_result <- node_type

# Now, the remaining entropy of each predictor w.r.t response variable needs to 
# be re-calculated considering the new column. 
# The response column and "chest_pain_type" predictor need not be included 
# for calculation. 

# Storing the response and "chest_pain_type" predictor column names in a vector.

excl_cols <- c(response_column,root_node_name)

# Storing the column index.

excl_cols_index <- which(colnames(heart_df_train_categorical) == excl_cols)

# Creating the column name with the exclusion.

column_name_vec_new <- colnames(heart_df_train_categorical[,-excl_cols_index])

# Column entropy and information gain vectors are created for storing the 
# remaining entropy and information gain respectively for each predictor. 

remaining_entropy_vec_new <- c()
info_gain_vec_new <- c()

# Calculation of remaining entropy and information gain.

for (column_name in column_name_vec_new){
  
  # Creating frequency table of response and predictor.
  
  freq_table <- table(heart_df_train_categorical[,column_name]
                      ,heart_df_train_categorical[,response_column])
  
  # Creating proportion table using frequency table.
  
  prop_freq_table <- prop.table(freq_table,margin = 1)
  
  # Creating the log-probability table. 
  
  log_table <- -1*prop_freq_table*log2(x = prop_freq_table)
  
  # If probability is zero, then log(probability) returns NA. 
  # That needs to be changed to 0, since multiplication with 0 is 0.
  
  for (col in colnames(log_table)){
    log_table[,col][is.nan(log_table[,col])] = 0
  }
  
  # The sum of probabilities needs to be calculated for each category in the 
  # predictor. This gives the overall proportion table of predictor vs response.
  
  sum_log_table <- rowSums(log_table)
  
  # Creating the feature proportion table from frequency table of response and predictor.
  
  feature_prop_table <- prop.table(rowSums(freq_table))
  
  # Calculation of remaining entropy.
  
  remaining_entropy <- sum(sum_log_table*feature_prop_table)
  
  # Calculation of information gain.
  
  Information_gain <- response_entropy-remaining_entropy
  
  # Storing the remaining entropy and information gain in their respective
  # vectors. 
  
  remaining_entropy_vec_new <- c(remaining_entropy_vec_new,remaining_entropy)
  
  info_gain_vec_new <- c(info_gain_vec_new,Information_gain)
}

# Assigning the names of the Column entropy and information gain vectors 
# with the entropy column name vector.

names(remaining_entropy_vec_new) <- column_name_vec_new
names(info_gain_vec_new) <- column_name_vec_new

# Printing the remaining entropy and information gain of all predictors.

for (column_index in 1:length(column_name_vec_new)){
  cat('The remaining entropy of predictor "'
      ,column_name_vec_new[column_index],'" is '
      ,remaining_entropy_vec_new[column_index],'.\n'
      ,sep = ''
      )
}

for (column_index in 1:length(column_name_vec_new)){
  cat('The information gain of predictor "'
      ,column_name_vec_new[column_index],'" is '
      ,info_gain_vec_new[column_index],'.\n'
      ,sep = ''
      )
}

# Printing the predictor that has the minimum remaining entropy, which also yields maximum
# information gain.

print(c(min(remaining_entropy_vec_new),names(which.min(remaining_entropy_vec_new))))
print(c(max(info_gain_vec_new),names(which.max(info_gain_vec_new))))

# Here, the new predictor "node_type_binary_result" has the highest information gain 
# when compared to the other predictors. 
# Plus, the information gain of "node_type_binary_result" predictor is almost same as
# that of the "chest_pain_type" predictor. This means similar information can
# be obtained from predictors "node_type_binary_result" or "chest_pain_type".

print(c(max(info_gain_vec_new),names(which.max(info_gain_vec_new))))
print(c(max(info_gain_vec),names(which.max(info_gain_vec))))

# This implies that a binary split could be used on the "node_type_binary_result"
# predictor. Binary split would simplify the tree structure and allow
# predictions with higher accuracy. There would be less chance for misclassification
# when are only two outcomes. When there are 4 outcomes, the chance for misclassification
# is bit higher. The prediction for the other levels in "chest_pain_type" predictor
# via binary split on the "node_type_binary_result" predictor is less complicated
# and could be done without much hurdle.

# Using a binary split on the continuous numeric predictor variables, 

# The original training data can be used here, as it already
# contains the numeric columns. The numerical data must be transformed into
# categorical data by splitting it into two. This can be done by finding the
# response category with the highest count and taking that data alone.
# The split will be done on the individual numeric columns on their
# corresponding mean in that data.

# The "dplyr" package will be used, as it is helpful for data manipulation.

# The package can be installed using: install.packages('dplyr')

library(dplyr)

# A frequency table with the response variable is created to identify which
# category has the highest count.

tbl_pred <- table(heart_df_train[,response_column])
print(c(max(tbl_pred),names(which.max(tbl_pred))))

# The training data with that highest count category is then stored separately.
# Here, response category "1" has the highest count of 116. 

heart_df_pred <- heart_df_train %>% filter(heart_disease_prediction == names(which.max(tbl_pred)))

# The mean for all the numeric data from the data set is stored in a vector.

mean_numeric_pred <- c()

for (numeric_col in numeric_columns){
  mean_numeric_pred <- c(mean_numeric_pred,mean(heart_df_pred[,numeric_col]))
}

# Assigning the names of the mean vectors with the column names.

names(mean_numeric_pred) <- numeric_columns

# Since the split in numerical data can lead to creation of many columns,
# the training data will be backed up and the numerical data will be replaced
# by its corresponding categorical data.

heart_df_train_bkp <- heart_df_train

# The binary chest pain type can also be added to the training data backup.

heart_df_train_bkp$node_type_binary_result <- node_type

for (numeric_col in numeric_columns){
  # The numeric column is to be removed and instead it will hold the categorical value.
  heart_df_train_bkp[,numeric_col] <- NULL
  
  # New column for the categorical data is created.
  new_col <- paste(numeric_col,'_cat',sep = '')
  
  # Mean for each numeric predictor is taken and compared with the data to get
  # the categorical value.
  mean_temp <- mean_numeric_pred[numeric_col]
  df_numeric <- heart_df_train[,numeric_col]
  
  # New column stores the categorical information.
  heart_df_train_bkp[,new_col] <- as.factor(ifelse(df_numeric > mean_temp,'Yes'
                                                   ,'No'
                                                   )
                                            )
}

# Now, the remaining entropy of each predictor w.r.t response variable needs to 
# be re-calculated considering the new columns. 
# The response column and "chest_pain_type" predictor need not be included 
# for calculation. 

# Creating the column name with the exclusion.

column_name_vec_num <- colnames(heart_df_train_bkp[,-excl_cols_index])

# Column entropy and information gain vectors are created for storing the 
# remaining entropy and information gain respectively for each predictor. 

remaining_entropy_vec_num <- c()
info_gain_vec_num <- c()

# Calculation of remaining entropy and information gain.

for (column_name in column_name_vec_num){
  
  # Creating frequency table of response and predictor. 9th column is response.
  
  freq_table <- table(heart_df_train_bkp[,column_name]
                      ,heart_df_train_bkp[,response_column])
  
  # Creating proportion table using frequency table.
  
  prop_freq_table <- prop.table(freq_table,margin = 1)
  
  # Creating the log-probability table. 
  
  log_table <- -1*prop_freq_table*log2(x = prop_freq_table)
  
  # If probability is zero, then log(probability) returns NA. 
  # That needs to be changed to 0, since multiplication with 0 is 0.
  
  for (col in colnames(log_table)){
    log_table[,col][is.nan(log_table[,col])] = 0
  }
  
  # The sum of probabilities needs to be calculated for each category in the 
  # predictor. This gives the overall proportion table of predictor vs response.
  
  sum_log_table <- rowSums(log_table)
  
  # Creating the feature proportion table from frequency table of response and predictor.
  
  feature_prop_table <- prop.table(rowSums(freq_table))
  
  # Calculation of remaining entropy.
  
  remaining_entropy <- sum(sum_log_table*feature_prop_table)
  
  # Calculation of information gain.
  
  Information_gain <- response_entropy-remaining_entropy
  
  # Storing the remaining entropy and information gain in their respective
  # vectors. 
  
  remaining_entropy_vec_num <- c(remaining_entropy_vec_num,remaining_entropy)
  
  info_gain_vec_num <- c(info_gain_vec_num,Information_gain)
}

# Assigning the names of the Column entropy and information gain vectors 
# with the entropy column name vector.

names(remaining_entropy_vec_num) <- column_name_vec_num
names(info_gain_vec_num) <- column_name_vec_num

# Printing the remaining entropy and information gain of all predictors.

for (column_index in 1:length(column_name_vec_num)){
  cat('The remaining entropy of predictor "'
      ,column_name_vec_num[column_index],'" is '
      ,remaining_entropy_vec_num[column_index],'.\n'
      ,sep = ''
      )
}

for (column_index in 1:length(column_name_vec_num)){
  cat('The information gain of predictor "'
      ,column_name_vec_num[column_index],'" is '
      ,info_gain_vec_num[column_index],'.\n'
      ,sep = ''
      )
}

# Printing the predictor that has the minimum remaining entropy, which also yields maximum
# information gain.

print(c(min(remaining_entropy_vec_num),names(which.min(remaining_entropy_vec_num))))
print(c(max(info_gain_vec_num),names(which.max(info_gain_vec_num))))

# Even after the inclusion of the continuous numeric predictors, the information
# gain of the predictor "node_type_binary_result" is the highest among all predictors.

print(c(max(info_gain_vec_num),names(which.max(info_gain_vec_num))))
print(c(max(info_gain_vec_new),names(which.max(info_gain_vec_new))))
print(c(max(info_gain_vec),names(which.max(info_gain_vec))))

# This implies that a binary split could be used on the "node_type_binary_result"
# predictor as it would simplify the tree structure and allow
# predictions with higher accuracy. There would be less chance for misclassification
# with only two outcomes. Hence, the "node_type_binary_result" predictor can
# be considered as the root node. The "node_type_binary_result" predictor is 
# the binary transformed data of "chest_pain_type" predictor in the original
# training data.

# Investigating the next level of split, i.e. which predictor 
# variable(s) should be used to split the first split found earlier. 
# Only binary splits are allowed again here. 

# The "node_type_binary_result" predictor is the root node element.
# It contains two values "Yes" and "No". As there are two values, the entropy and
# information gain must be calculated for the training data corresponding to 
# those values. Which means that the entropy and information gain will be calculated twice.
# One for training data with the "Yes" value in "node_type_binary_result" predictor 
# and the other with the "No" value. 

# In the training data, the below columns hold 2 types of data (binary results): 

# "sex"
# "fasting_blood_sugar"
# "exercise_induced_angina"
# "age_cat"
# "resting_blood_pressure_cat"
# "serum_cholestoral_cat"
# "max_heart_rate_achieved_cat"
# "oldpeak_cat"

# Below are the remaining entropy and information gain values.

# Remaining Entropy:-
# -------------------

# "sex": 0.9313673.
# "fasting_blood_sugar": 0.9953725.
# "exercise_induced_angina": 0.8802547.
# "age_cat": 0.9547184.
# "resting_blood_pressure_cat": 0.9899705.
# "serum_cholestoral_cat": 0.9781485.
# "max_heart_rate_achieved_cat": 0.9294486.
# "oldpeak_cat": 0.8812828.

# Information Gain:-
# ------------------

# "sex": 0.06467109.
# "fasting_blood_sugar": 0.0006658582.
# "exercise_induced_angina": 0.1157837.
# "age_cat": 0.04131992.
# "resting_blood_pressure_cat": 0.006067855.
# "serum_cholestoral_cat": 0.01788988.
# "max_heart_rate_achieved_cat": 0.06658976.
# "oldpeak_cat": 0.1147556.

# Since a binary split must be done, the categorical data that hold more than 2
# levels, also need to be transformed to hold 2 levels. The approach is like d) 
# and should be done for all the columns that hold more than 2 levels.
# On this dataset, the remaining entropy and information gain should be carried 
# out to find the terminal nodes. This would be the next level of split 
# with binary outcomes.

# The categorical data that hold more than 2 needs to be transformed to hold
# 2 levels. As there could be creation of many columns, training data is backed up.

heart_df_train_bkp_cat <- heart_df_train_bkp

# "chest_pain_type" category could be removed from the dataset.

heart_df_train_bkp_cat[,root_node_name] <- NULL

# Creating the column name vector. Column "chest_pain_type" is excluded.

root_node_index <- which(colnames(heart_df_train_bkp) == root_node_name)

col_name_vec <- colnames(heart_df_train_bkp)[-root_node_index]

for (column_name in col_name_vec){
  
  # Check if a column holds more than 2 levels of data.
  
  if (length(levels(heart_df_train_bkp_cat[,column_name]))>2){
    
    # The original column is to be removed and instead it will hold the 
    # binary categorical value.
    
    heart_df_train_bkp_cat[,column_name] <- NULL
    
    # New column for the categorical data is created.
    
    new_col <- paste(column_name,'_bin',sep = '')
    
    # The mode (most occurred value) of the predictor is taken and compared with
    # the data to get the categorical value.
    
    name_max <- names(which.max(table(heart_df_train_bkp[,column_name])))
    
    # New column stores the categorical information.
    
    df_bin_cat <- heart_df_train_bkp[,column_name]
    
    heart_df_train_bkp_cat[,new_col] <- as.factor(ifelse(df_bin_cat == name_max,'Yes','No'))
  }
}

# The root node and its levels of data are stored in separate variables.

root_node_column <- names(which.max(info_gain_vec_num))

root_node_factor <- factor(levels(heart_df_train_bkp_cat[,root_node_column]))

# For every value in the root node, the Remaining Entropy and information gain
# must be calculated with respect to the training data. 
# These values could be stored in separate lists.

remaining_entropy_list <- list()

information_gain_list <- list()

# The response variable need not be included for calculation. 

response_index <- which(colnames(heart_df_train_bkp_cat) == response_column)

column_name_vec_node <- colnames(heart_df_train_bkp_cat[,-response_index])

for (node_level in root_node_factor){
  
  cat('Calculation of remaining entropy and information gain '
      ,'for the node level value = ',node_level,'\n\n',sep = '')
  
  # Creating a dataset for processing individual values in root node.
  
  heart_df_node <- heart_df_train_bkp_cat %>% filter(node_type_binary_result == node_level)
  
  # Column entropy and information gain vectors are created for storing the 
  # remaining entropy and information gain respectively for each predictor. 
  
  remaining_entropy_vec_node <- c()
  info_gain_vec_node <- c()
  
  # Calculation of remaining entropy and information gain.
  
  for (column_name in column_name_vec_node){
    
    # Creating frequency table of response and predictor.
    
    freq_table <- table(heart_df_node[,column_name],heart_df_node[,response_index])
    
    # Creating proportion table using frequency table.
    
    prop_freq_table <- prop.table(freq_table,margin = 1)
    
    # Creating the log-probability table. 
    
    log_table <- -1*prop_freq_table*log2(x = prop_freq_table)
    
    # If probability is zero, then log(probability) returns NA. 
    # That needs to be changed to 0, since multiplication with 0 is 0.
    
    for (col in colnames(log_table)){
      log_table[,col][is.nan(log_table[,col])] = 0
    }
    
    # The sum of probabilities needs to be calculated for each category in the 
    # predictor. This gives the overall proportion table of predictor vs response.
    
    sum_log_table <- rowSums(log_table)
    
    # Creating the feature proportion table from frequency table of response and predictor.
    
    feature_prop_table <- prop.table(rowSums(freq_table))
    
    # Calculation of remaining entropy.
    
    remaining_entropy <- sum(sum_log_table*feature_prop_table)
    
    # Calculation of information gain.
    
    Information_gain <- response_entropy-remaining_entropy
    
    # Storing the remaining entropy and information gain in their respective
    # vectors. 
    
    remaining_entropy_vec_node <- c(remaining_entropy_vec_node,remaining_entropy)
    
    info_gain_vec_node <- c(info_gain_vec_node,Information_gain)
  }
  
  # Assigning the names of the Column entropy and information gain vectors 
  # with the entropy column name vector.
  
  names(remaining_entropy_vec_node) <- column_name_vec_node
  names(info_gain_vec_node) <- column_name_vec_node
  
  # Storing the remaining entropy and information gain in their respective lists.
  
  remaining_entropy_list[[node_level]] <- remaining_entropy_vec_node
  information_gain_list[[node_level]] <- info_gain_vec_node
  
  # Printing the remaining entropy and information gain of all predictors.
  
  for (column_index in 1:length(column_name_vec_node)){
    cat('The remaining entropy of predictor "'
        ,column_name_vec_node[column_index],'" is '
        ,remaining_entropy_vec_node[column_index],'.\n'
        ,sep = ''
        )
  }
  cat('\n')
  for (column_index in 1:length(column_name_vec_node)){
    cat('The information gain of predictor "'
        ,column_name_vec_node[column_index],'" is '
        ,info_gain_vec_node[column_index],'.\n'
        ,sep = ''
        )
  }
  
  # Printing the predictor that has the minimum remaining entropy, which also yields maximum
  # information gain.
  
  cat('\n')
  print(c(min(remaining_entropy_vec_node),names(which.min(remaining_entropy_vec_node))))
  print(c(max(info_gain_vec_node),names(which.max(info_gain_vec_node))))
  cat('\n')
  
}

# Printing the predictor that yields the maximum information gain for each node level.

for (node_level in root_node_factor){
  
  # Storing the remaining entropy and information gain in vectors for each node level.
  
  rem_entr_node <- remaining_entropy_list[[node_level]]
  info_gain_node <- information_gain_list[[node_level]]
  
  print(c("Node level:",node_level))
  print(c(min(rem_entr_node),names(which.min(rem_entr_node))))
  print(c(max(info_gain_node),names(which.max(info_gain_node))))
  cat('\n')
}

# For the "No" value in "node_type_binary_result" predictor, the terminal node
# would be the binary predictor "slope_peak_exercise_ST_segment_bin", which is the
# transformed version of the "slope_peak_exercise_ST_segment" predictor.

# For the "Yes" value in "node_type_binary_result" predictor, the terminal node
# would be the binary predictor "number_of_major_vessels_bin", which is the
# transformed version of the "number_of_major_vessels" predictor.

# Predicting the class for the test set using if/subsetting. 

# To predict the class for the test data set, the root node and 
# terminal node columns present in the test data set needs to be transformed
# to hold binary values as done before for the training datasets.

# Terminal nodes can be stored in a separate vector.

terminal_node_vec <- c()

for (node_level in root_node_factor){
  
  # Name of the terminal node.
  
  terminal_node <- names(which.max(information_gain_list[[node_level]]))
  
  # Storing the terminal node in the vector.
  
  terminal_node_vec <- c(terminal_node_vec,terminal_node) 
}  

# Assigning the names to the terminal node vector.

names(terminal_node_vec) <- root_node_factor

# Creating a backup for the test data.

heart_df_test_bkp <- heart_df_test

# Creation of a column name vector for storing the original root and terminal nodes,
# that is, the names before the binary categorical transformation.

# It is better to trim the '_bin' string in the "terminal_node_vec" vector to get the
# original vector. The '_bin' string was added to the distinguish b/w the original
# categorical data and the transformed data. It's done using the trimws function.

original_terminal_node_vec <- trimws(terminal_node_vec
                                     ,which = c("both", "left", "right")
                                     ,whitespace = "_bin"
                                     )

# Creating a vector that holds the binary root and terminal nodes.

node_vec <- c(root_node_column,terminal_node_vec)

# Creating a vector that holds the original root and terminal nodes.

original_node_vec <- c(root_node_name,original_terminal_node_vec)
  
# Creation of the binary root node and terminal node columns in the test data set.

for (node_column_index in 1:length(original_node_vec)){
  
  # Store the node column name in a variable.
  
  node_column_name <- original_node_vec[node_column_index]
  
  # Check if a column holds more than 2 levels of data.
  
  if (length(levels(heart_df_test_bkp[,node_column_name]))>2){
    
    # New column for the categorical data is created.
    
    new_col <- node_vec[node_column_index]
    
    # The mode of the predictor is taken and compared with the data to get
    # the categorical value.
    
    name_max <- names(which.max(table(heart_df_test[,node_column_name])))
    
    # New column stores the categorical information.
    
    df_bin_cat <- heart_df_test[,node_column_name]
    
    heart_df_test_bkp[,new_col] <- as.factor(ifelse(df_bin_cat == name_max,'Yes','No'))
  }
}

# Once the root and terminal nodes are created, those values should be mapped to
# the training data set. For each row, the root and terminal node value is 
# mapped with the training data, and then the response value with the highest 
# probability/count will be taken and stored in a new column. 
# This would be considered as the predicted value.

# Creating the new column for predicted response.

response_column_new <- paste(response_column,'_observed',sep = '')

# Creating a temporary vector to store the response value with the highest count.

response_max_prob <- c()

for (row_index in 1:nrow(heart_df_test_bkp)){
  
  # Storing the individual row of the test data in a variable.
  
  heart_df_test_row <- heart_df_test_bkp[row_index,]
  
  # Storing the root node value for the row.
  
  test_root_node <- heart_df_test_row[,root_node_column]
  
  # Storing the column name of the terminal node for the row.
  
  terminal_node <- terminal_node_vec[heart_df_test_row[,root_node_column]]
  
  # Storing the terminal node value for the row.
  
  test_terminal_node <- heart_df_test_row[,terminal_node]
  
  # Storing the training data for the terminal node.
  
  train_terminal_node <- heart_df_train_bkp_cat[,terminal_node] 
  
  # Storing the training data for the root node.
  
  train_root_node <- heart_df_train_bkp_cat[,root_node_column] 
  
  # Storing the training data for the root and terminal node values.
  
  df_root_term <- heart_df_train_bkp_cat %>% filter(train_terminal_node == test_terminal_node
                                                    & train_root_node == test_root_node
                                                    )
  
  # Creating frequency table for the response column.
  
  freq_table_response <- table(df_root_term[,response_column])
  
  # Storing the response withe most likely prediction value in the temporary vector.
  
  response_max_prob <- c(response_max_prob,names(which.max(freq_table_response)))
}

# "Root Node factor: No"
# "Terminal node column:" "slope_peak_exercise_ST_segment_bin" 
# "Terminal Node Value: " "No"             
# "heart_disease_prediction (Most Likely):" "1"                        
# "Terminal Node value: " "Yes"            
# "heart_disease_prediction (Most Likely): " "1"                        

# "Root Node factor: Yes"
# "Terminal node column:" "number_of_major_vessels_bin" 
# "Terminal Node Value: " "No"             
# "heart_disease_prediction (Most Likely):" "2"                        
# "Terminal Node Value: " "Yes"            
# "heart_disease_prediction (Most Likely):" "2"  

# Storing the value in the temporary vector in the new column of the test dataset.

heart_df_test_bkp[,response_column_new] <- as.factor(response_max_prob)

# Creation of confusion matrix.

heart_df_conf_matr <- table(heart_df_test_bkp[,response_column_new]
                            ,heart_df_test_bkp[,response_column]
                            )

# False positive rate is the number of false positive entries divided by the 
# sum of number of false positive and true negative entries.

fp_rate <- heart_df_conf_matr[1,2]/sum(heart_df_conf_matr[,2])

cat('The False positive rate is ',100*fp_rate,'%.\n',sep = '')

# The False positive rate is 25%.

# True positive rate is the number of true positive entries divided by the 
# sum of number of true positive and false negative entries.

tp_rate <- heart_df_conf_matr[1,1]/sum(heart_df_conf_matr[,1])

cat('The True positive rate is ',100*tp_rate,'%.\n',sep = '')

# The True positive rate is 67.64%.

# Precision is the number of true positive entries divided by the 
# sum of number of true positive and false positive entries.

precision <- heart_df_conf_matr[1,1]/sum(heart_df_conf_matr[1,])

cat('The precision is ',100*precision,'%.\n',sep = '')

# The precision is 82.14%.

# Recall is the number of true positive entries divided by the 
# sum of number of true positive and false negative entries.

recall <- heart_df_conf_matr[1,1]/sum(heart_df_conf_matr[,1])

cat('Recall is ',100*recall,'%.\n',sep = '')

# The recall is 67.64%.

# F-measure is 2 divided by the sum of reciprocals of precision and recall values.

f_measure <- 2/((1/precision)+(1/recall))

cat('F-measure is ',100*f_measure,'%.\n',sep = '')

# F-measure is 74.19355%.

# Baseline rate is the probability of the response value with the highest count.

baseline_rate <- max(prop.table(table(heart_df_train[,response_column])))

cat('The baseline rate is ',baseline_rate,'.\n',sep = '')

# Overall accuracy is the sum of the diagonals in the confusion matrix divided
# by the number of rows in the test data set.

overall_accuracy <- sum(diag(heart_df_conf_matr))/nrow(heart_df_test_bkp)

cat('The overall accuracy is ',overall_accuracy,'.\n',sep = '')

s <- paste('This is a good model, since overall accuracy ('
           ,round(overall_accuracy,2),') is greater than the baseline rate ('
           ,round(baseline_rate,2),').',sep = ''
           )

if (overall_accuracy < baseline_rate){
  
  s <- paste('This is not a good model, since overall accuracy ('
             ,round(overall_accuracy,2),') is lesser than the baseline rate ('
             ,round(baseline_rate,2),').',sep = ''
             )
}
  
print(s)

# This is a good model, since overall accuracy (0.7) is greater than 
# the baseline rate (0.54). This indicates that the splitting criteria used for
# creating the root & terminal nodes of a decision tree is correct. This approach
# could be used for extension of the terminal nodes, which in turn can create an
# accurate tree model.

# Using the tree function from the package tree to build a decision 
# tree and comparing the results obtained previously 

# The "tree" package can be installed using: install.packages('tree').

# Invoking the "tree" package.

library(tree)

# Using the tree function to build a decision tree of the training data set.

# The heart "heart_disease_prediction" is the response column, so it is excluded
# from the predictor variables.

# Creating the decision tree, with the subset as the training rows.

tree_heart <- tree(heart_disease_prediction~.-heart_disease_prediction
                   ,heart_df
                   ,subset = heart_df_train_rows
                   )

# Plotting the decision tree.

plot(tree_heart)

text(tree_heart, pretty = 1)

# From the tree diagram, it can be said that the root node is "chest_pain_type"
# and the terminal nodes in the first split are "slope_peak_exercise_ST_segment"
# and "number_of_major_vessels" for "chest_pain_type" values 1,2,3 and 4 respectively.

# This is identical to the results obtained in e) and g).
# The root node column "node_type_binary_result" is the binary version of the original root node 
# "chest_pain_type" and the terminal nodes "slope_peak_exercise_ST_segment_bin"   
# and "number_of_major_vessels_bin" are the binary versions of the original 
# terminal nodes "slope_peak_exercise_ST_segment" and "number_of_major_vessels".

# If "chest_pain_type" value is "4", then the "node_type_binary_result" is "Yes",
# else the "node_type_binary_result" is "No". This is because value "4" has the highest
# chance of occurrence.

# For the "No" value in "node_type_binary_result" predictor, the terminal node
# would be the binary predictor "slope_peak_exercise_ST_segment_bin".

# For the "Yes" value in "node_type_binary_result" predictor, the terminal node
# would be the binary predictor "number_of_major_vessels_bin".

# This is verified by plotting the decision tree for the transformed training data.

tree_heart_train_cat <- tree(heart_disease_prediction~.-heart_disease_prediction
                             ,heart_df_train_bkp_cat)

# Plotting the decision tree.

plot(tree_heart_train_cat)

text(tree_heart_train_cat, pretty = 1)

# predict() function is used for making predictions from the results of various
# model fitting functions.

# Predicting the response value from the tree using the test data set.

tree_heart_pred <- predict(tree_heart,heart_df_test,type = "class")

# Creation of confusion matrix using the decision tree.

heart_df_conf_matr_tree <- table(tree_heart_pred,heart_df_test[,response_column])

# Obtaining the overall accuracy from the confusion matrix.

overall_accuracy_tree <- sum(diag(heart_df_conf_matr_tree))/nrow(heart_df_test)

cat('The overall accuracy of the tree is ',overall_accuracy_tree,'.\n',sep = '')

s <- paste('Calculated overall accuracy (',round(overall_accuracy,2)
           ,') is greater than the overall accuracy ('
           ,round(overall_accuracy_tree,2),') from the tree.',sep = ''
           )

if (overall_accuracy < overall_accuracy_tree){
  
  s <- paste('Calculated overall accuracy (',round(overall_accuracy,2)
             ,') is lesser than the overall accuracy ('
             ,round(overall_accuracy_tree,2),') from the tree.',sep = ''
             )
}

print(s)

# Calculated overall accuracy (0.7) is lesser than the overall accuracy (0.85) 
# from the tree. The difference is due to the algorithms used in the tree creation.
# The tree () function uses 'Gini' indexing as splitting criteria, while the calculated
# one uses Entropy and information gain. 

summary(tree_heart)

# The tree summary shows that there are 19 terminal nodes. This is a huge tree and
# there is a strong chance for misclassification. Hence, this tree could be pruned
# to check if there is any increase in model accuracy. It can be done using cross-fold
# validation techniques. Here, 19 is the best cross-validation error rate.

# Setting the seed to control the randomness.

set.seed(750)

cv_tree_heart <- cv.tree(tree_heart,K = 5, FUN = prune.misclass)

# Here 5 folds of cross-validation is used.

# Plots of deviance vs size and deviance vs 'k' for the cross validated tree.

par(mfrow = c(1,2))

plot(x = cv_tree_heart$size,y = cv_tree_heart$dev,type = "b"
     ,main = 'Plot of deviance vs size',xlab = 'size',ylab = 'deviance'
     )

plot(x = cv_tree_heart$k,y = cv_tree_heart$dev,type = "b"
     ,main = 'Plot of deviance vs k',xlab = 'k',ylab = 'deviance'
     )

# Finding the size which holds the minimum deviance. 

cv_best_tree_size <- cv_tree_heart$size[which.min(cv_tree_heart$dev)]

cat('Size of tree = ',cv_best_tree_size,' looks best here.\n',sep = '')

# Size of tree = 6 looks best here.

# Creating the pruned tree with the best size = 6.

prune_tree_heart <- prune.misclass(tree_heart,best = cv_best_tree_size)

# Plotting the pruned tree.

par(mfrow = c(1,1))

plot(prune_tree_heart)

text(prune_tree_heart,pretty = 1)

# Predicting the response value from the pruned tree using the test data set.

prune_tree_heart_pred <- predict(prune_tree_heart,heart_df_test,type = "class")

# Creation of confusion matrix using the pruned tree.

heart_df_conf_matr_prune_tree <- table(prune_tree_heart_pred
                                       ,heart_df_test[,response_column])

# Obtaining the overall accuracy from the confusion matrix.

overall_accuracy_prune_tree <- sum(diag(heart_df_conf_matr_prune_tree))/nrow(heart_df_test)

cat('The overall accuracy of the pruned tree is '
    ,overall_accuracy_prune_tree
    ,'.\n'
    ,sep = ''
    )

s <- paste("Pruned tree's overall accuracy ("
           ,round(overall_accuracy_prune_tree,2)
           ,") is greater than the original decision tree's overall accuracy ("
           ,round(overall_accuracy_tree,2),').',sep = ''
           )

if (overall_accuracy_prune_tree < overall_accuracy_tree){
  
  s <- paste("Pruned tree's overall accuracy ("
             ,round(overall_accuracy_prune_tree,2)
             ,") is lesser than the original decision tree's overall accuracy ("
             ,round(overall_accuracy_tree,2),').',sep = ''
             )
}

print(s)

# The test set error (overall accuracy) of the tree is 87%.

# Pruning the tree is a better option since the accuracy has increased from 
# 85% to 87%. This would make a better model with more accurate predictions.

# Improving the results by using a random forest model. 

# The main drawback of using decision trees is that they produce high variance.
# However, having models with low variance is necessary. Hence, bagging aka 
# bootstrapping aggregation is to be used. Bootstrapping is re-sampling the observed 
# data by randomly sampling (within itself) with replacement. The sample length will 
# be same as that of the observation length. Due to this, plenty of training datasets 
# would be created, which can then be averaged to reduce the variance. 
# The trees are not pruned here, which means that each individual tree has high 
# variance but low bias (because of random sampling). Averaging this reduces both the 
# variance and the bias. The remaining data from the randomly selected data could
# be considered as the test data. This data is also known as Out-of-Bag data (OOB).
# The MSE of the OOB error is calculated and the model with lowest MSE is considered.

# For creating random forests, the "randomForest" package is needed. 
# It can be installed using: install.packages('randomForest').

# Invoking the "randomForest" package.

library(randomForest)

# Setting the seed to control the randomness.

set.seed(750)

# Using the random forest algorithm on the training data set. Number of trees = 1000.
# Here mtry = root of the number of predictors.

rf_heart_train <- randomForest(formula = heart_disease_prediction~.-heart_disease_prediction
                               ,data = heart_df
                               ,subset = heart_df_train_rows
                               ,ntree = 1000
                               ,importance = TRUE
                               ,do.trace = TRUE
                               )
print(rf_heart_train)

# Plotting the random forest model.

plot(rf_heart_train,main = 'Plot of the random forest model')

# Storing the confusion matrix from the random forest to a variable.

heart_df_accuracy_conf_matr_rf <- rf_heart_train$confusion

accuracy_score_rf <- sum(diag(heart_df_accuracy_conf_matr_rf))/nrow(heart_df_train)

cat('The accuracy score is ',100*accuracy_score_rf,'%.\n',sep = '')

# The accuracy score is 83.33333%.

# Predicting the response value from the random forest using the test data set.

rf_heart_pred <- predict(rf_heart_train,heart_df_test,type = "class")

# Creation of confusion matrix using the decision tree.

heart_df_conf_matr_rf <- table(rf_heart_pred,heart_df_test[,response_column])

# Obtaining the overall accuracy from the confusion matrix.

overall_accuracy_rf <- sum(diag(heart_df_conf_matr_rf))/nrow(heart_df_test)

cat('The overall accuracy of the random forest model is '
    ,overall_accuracy_rf
    ,'.\n'
    ,sep = ''
    )

s <- paste("Random forest's overall accuracy (",round(overall_accuracy_rf,2)
           ,") is greater than the pruned tree's overall accuracy ("
           ,round(overall_accuracy_prune_tree,2),').',sep = ''
           )

if (overall_accuracy_rf < overall_accuracy_prune_tree){
  
  s <- paste("Random forest's overall accuracy (",round(overall_accuracy_rf,2)
             ,") is lesser than the pruned tree's overall accuracy ("
             ,round(overall_accuracy_prune_tree,2),').',sep = ''
  )
}

print(s)

# Random forest's overall accuracy (0.85) is lesser than the pruned tree's 
# overall accuracy (0.87). The accuracy score (83.33) is also lesser than the pruned tree's 
# overall accuracy (0.87). This means that the random forest technique need not 
# be applied, as pruning is much more accurate.
