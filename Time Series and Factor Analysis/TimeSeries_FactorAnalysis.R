# 1. TIME SERIES ANALYSIS

# Analysis of the Time Series “AirPassengers” in RStudio. 
# Use the libraries available in R which you consider are appropriate for this study.
# (Hint: you can make use of “astsa”, “forecast” and “tseries” for the analysis).

# Preliminar analysis

# Question:- 

# • Carry out a preliminary descriptive analysis on the data. 
# Use summary statistics and plots to describe the dataset.

# Answer:-

# Loading the libraries required for Time Series Analysis.

library(astsa)
library(forecast)
library(tseries)
library(foreign)
library(summarytools)
library(graphics)

# Creating a replica of the “AirPassengers” dataset. This replica would be used
# for analysis, model building and forecasts.

df_AirPassengers <- AirPassengers

# Viewing the dataset.

print(df_AirPassengers)

# Check if the dataset contains missing value. If so, we need to handle them.

sum(is.na(df_AirPassengers))

# There are no missing values in the dataset.

# Checking the frequency of the dataset using the cycle.

cycle(df_AirPassengers)

frequency(df_AirPassengers)

# From the cycle, we can say that the frequency is 12.

# Viewing the summary of the dataset.

summary(df_AirPassengers)

# The minimum number of passengers is 104 and the maximum is 622.

# Visualising the data using autoplot () function.

autoplot(df_AirPassengers, main = "Plot of the Air Passengers dataset") 

# From the plot, we can infer that the number of passengers increases over
# time for each year. There appears to be an upward linear trend owing to the
# high demand for air travel. This dataset seems to be a multiplicative time 
# series due to the increase in seasonality over the years.

# Visualising the data using a boxplot to check for seasonal impacts.

boxplot(df_AirPassengers~cycle(df_AirPassengers),xlab = 'Month'
        ,ylab = "Number of passengers (1000's)"
        ,main = 'Boxplot of monthly passengers from early 1949 to end of 1960'
        ,col = rainbow(frequency(df_AirPassengers)))

# From the boxplot we can infer that the number of passengers is quite high 
# in the months of June, July, August and September. These observations possess
# the highest variability in the data. This stipulates that there is a seasonality
# over a cycle of 12 months (1 year). This also means that the majority of the
# people travel by air to other places during the summer season to enjoy their
# vacation. There are no outliers or missing values in the dataset which means
# that there is no need for data cleaning.

# Question:- 

# • Decompose the dataset and analyze the different components of the series. 
# State what type of model – additive or multiplicative – is more appropriate to
# describe this series. Help your conclusions by using appropriate plots and summaries.

# Answer:-

# Decomposing the dataset into additive and multiplicative components.

decom_add_AP <- decompose(df_AirPassengers, "additive")

decom_mul_AP <- decompose(df_AirPassengers, "multiplicative")

# Visualizing the additive and multiplicative components.

autoplot(decom_add_AP)

autoplot(decom_mul_AP)

# From the decomposed plots we can observe the linear increase in trend with the
# seasonality. However, it is difficult to assess the decomposition type just
# by observing the plot of the components. We need to find the correlation  
# between the data and the residuals (random component). This can be done by
# applying the acf () function on the random components. Since there could be 
# negative values, we find the sum of squares of the acf from both the models. 
# The model type with the least value is the more appropriate.

ss_acf_add <- sum(acf(decom_add_AP$random,na.action = na.omit)$acf^2)

ss_acf_mul <- sum(acf(decom_mul_AP$random,na.action = na.omit)$acf^2)

s_ss_acf <- paste('Since the sum of squares of Acf of additive model ('
                  ,ss_acf_add,') is lesser than the multiplicative model ('
                  ,ss_acf_mul, '), the additive model is more appropriate.'
                  ,sep = '')

if (ss_acf_add > ss_acf_mul){
  
  s_ss_acf <- paste('Since the sum of squares of Acf of multiplicative model ('
                    ,ss_acf_mul,') is lesser than the additive model ('
                    ,ss_acf_add, '), the multiplicative model is more appropriate.'
                    ,sep = '')
  
}

print(s_ss_acf)

# Since the sum of squares of Acf of multiplicative model (1.918) is lesser 
# than the additive model (4.0847), the multiplicative model is more appropriate.

# Question:- 

# • Analyse which of the components (if any) has a bigger impact on the time series. 
# Is this component conditioning the analysis in any way? Explain how this issue
# could be addressed in case it would be required.

# Answer:-

# Storing the plot title in a string variable.

str_plt_title <- "Multiplicative Time Series Plot of Air Passengers dataset"

stl(df_AirPassengers,s.window = 12) %>% autoplot(main = str_plt_title)

# From the plot, we can see that the trend component has the highest impact on 
# the time series since it has the highest variability among all the components. 
# Too much variability in trend will impact future predictions and it has to be 
# addressed. This can be done by first applying log transformations on the 
# dataset to reduce the variability and then taking the first difference of the
# log transformed values. This will remove the variability and trend.

# Question:- 

# • Indicate if you have found periodicity on the time series.

# Answer:-

# Periodicity could be found using the mstl () function.

autoplot(mstl(df_AirPassengers),main = str_plt_title)

# The Periodicity is 12 from the label of the Seasonal component.

# Time series modelling

# Question:- 

# • Implement the Classical Method (SES, Holt’s, Holt-Winter’s) that you consider 
# is more appropriate for describing this data and explain why do you think it 
# is the best option. Implement your chosen model using both model options 
# (additive and multiplicative) and explain which of the models do you think is 
# better for this time series. Support your conclusions with the use of GOF measures.
# Is this conclusion consistent with what you expected in the preliminary analysis?

# Answer:-

# Simple Exponential Smoothing (SES) is used for time series that have neither
# trend nor seasonality. Double Exponential Smoothing is used for time series 
# that have trend but no seasonal component. Triple Exponential Smoothing is 
# used for time series that have both the trend and seasonal components.
# Since the time series plot of Air Passengers dataset has both the components,
# we can implement Triple Exponential Smoothing models such as Holt-Winter's model.

# Implementing Holt-Winter's additive and multiplicative models.

par(mfrow = c(1,2))

hw_add_AP <- hw(df_AirPassengers, seasonal = "additive")

plot(hw_add_AP)

hw_mul_AP <- hw(df_AirPassengers, seasonal = "multiplicative")

plot(hw_mul_AP)

par(mfrow = c(1,1))

# The forecast from the multiplicative model appears to be better than the
# additive model. However, we need to confirm that using Goodness of Fit (GOF)
# measures. The GOF of Holt-Winter's model is determined by Akaike’s Information
# Criterion (AIC), Akaike’s Corrected Information Criterion (AICC) and Bayesian 
# Information Criterion (BIC). When comparing two models using AIC, the model 
# with the lower AIC value is the better model.

s_aic <- paste("Since the AIC of Holt-Winter's additive model ("
               ,hw_add_AP$model$aic
               ,") is lesser than the AIC of Holt-Winter's multiplicative model ("
               ,hw_mul_AP$model$aic, '), the additive model is more appropriate.'
               ,sep = '')
  
if (hw_add_AP$model$aic > hw_mul_AP$model$aic){
  
  s_aic <- paste("Since the AIC of Holt-Winter's multiplicative model ("
                 ,hw_mul_AP$model$aic
                 ,") is lesser than the AIC of Holt-Winter's additive model ("
                 ,hw_add_AP$model$aic
                 , '), the multiplicative model is more appropriate.',sep = '')
  
}

print(s_aic)

# Since the AIC of Holt-Winter's multiplicative model (1405.654) is lesser than 
# the AIC of Holt-Winter's additive model (1565.871), the multiplicative model 
# is more appropriate. This conclusion is consistent with what was expected in the
# preliminary analysis.

# Question:- 

# • Indicate whether this time series is stationary. Use the formal tests and 
# graphical tools you consider appropriate to support your conclusion.
# In case it is not stationary, manipulate the dataset to obtain an appropriate 
# series which meets the stationarity conditions for further analysis using 
# ARIMA models. Explain and justify all the process in the transformation.
# (Hint: you need to find any possible issues with trend, variability, seasonality 
# and periodicity on the dataset)

# Use the correlograms for analyzing the transformed time series. 
# Based on the correlograms, what type of ARIMA-SARIMA model do you expect to 
# describe this dataset. Explain your answer.

# Chose 3 models you think are good candidates for this dataset and analyze the 
# residuals. Use plots and formal tests to compare the models.

# Find the model proposed by R using “auto.arima” command and discuss whether 
# you think this is a good option for this dataset. Do you think that any of 
# your models would be a better option?

# Answer:-

# Stationarity is an important pre-requisite for building ARIMA models.

# Augmented Dickey-Fuller (ADF) Test can be used for testing the stationarity of
# a time series. 

# H0: Time Series is not stationary.
# HA: There is stationarity in the time series.

adf.test(df_AirPassengers, alternative = "stationary")

# Since p-value (0.01) is less than alpha (0.05), we reject the null hypothesis.
# This means that there is no evidence of non-stationarity.

# However, we need to confirm that using Kwiatkowski-Phillips-Schmidt-Shin (KPSS) 
# test. This is test is used for checking the level/trend stationarity of a time
# series.

# H0: Time Series is level/trend stationary.
# HA: Time Series is not stationary.

kpss.test(df_AirPassengers)

# Since p-value (0.01) is less than alpha (0.05), we reject the null hypothesis.
# This means that there is no evidence of level/trend stationarity. 
# This result is inconsistent from the ADF test result. Hence, we must transform
# the data to induce stationarity.

# We can apply log transformations to remove the variability.

log_df_AP <- log(df_AirPassengers)

autoplot(log_df_AP, main = "Log transformation on Air Passengers Data")

# Storing the plot title in a string variable.

str_plt_title_log <- paste("Multiplicative Time Series Plot of"
                           ,"Log Transformed","Air Passengers dataset")

stl(log_df_AP,s.window = 12) %>% autoplot(main = str_plt_title_log)

# The seasonality appears to have almost constant variability.

# We perform the ADF and KPSS tests on the log transformed data.

adf.test(log_df_AP, alternative = "stationary")

# Since p-value (0.01) is less than alpha (0.05), we reject the null hypothesis.
# This means that there is no evidence of non-stationarity after log transformation.

kpss.test(log_df_AP)

# Since p-value (0.01) is less than alpha (0.05), we reject the null hypothesis.
# This means that there is no evidence of level/trend stationarity. 
# This result is inconsistent from the ADF test result. Hence, we must transform
# the data once more to induce stationarity.

# We apply a first difference transform to remove the trend.

diff_log_df_AP <- diff(log_df_AP)

autoplot(diff_log_df_AP, main = paste("First difference Transformation on the"
                                      ,"Log transformed data"))

# We can see that the plot almost resembles random noise.

# Storing the plot title in a string variable.

str_plt_title_log_diff <- paste("Multiplicative Time Series Plot of first"
                                ,"differenced Log Transformed"
                                ,"Air Passengers dataset")

autoplot(decompose(diff(log_df_AP),'multiplicative'),main = str_plt_title_log_diff)

# We can see that there is no pattern in the trend and variability in the 
# seasonality has been removed. The time series appears to be stationary.
# However, we need to perform the ADF and KPSS tests to confirm this.

# We perform the ADF and KPSS tests on the first differenced log transformed data.

adf.test(diff_log_df_AP, alternative = "stationary")

# Since p-value (0.01) is less than alpha (0.05), we reject the null hypothesis.
# This means that there is no evidence of non-stationarity after first 
# differenced log transformation.

kpss.test(diff_log_df_AP)

# Since p-value (0.1) is more than alpha (0.05), we fail to reject the 
# null hypothesis. This means that there is no evidence to show that the 
# transformed time series is non-stationary. This result is consistent with the 
# ADF test result. However, the seasonality drastically dominates the time series.
# Hence, we need to perform second differencing to eliminate this.

# We apply a second difference transform to remove the seasonality.

diff2_log_df_AP <- diff(diff_log_df_AP,lag = 1,differences = 12)

autoplot(diff2_log_df_AP, main = paste("Second difference Transformation on the"
                                       ,"Log transformed data"))

# We can see that the plot closely resembles random noise.

# Storing the plot title in a string variable.

str_plt_title_log_diff2 <- paste("Multiplicative Time Series Plot of second"
                                 ,"differenced Log Transformed"
                                 ,"Air Passengers dataset")

autoplot(decompose(diff2_log_df_AP,'multiplicative'),main = str_plt_title_log_diff2)

# We can see that the trend resembles white noise and the time series appears to 
# be almost stationary. However, we need to perform the ADF and KPSS tests to 
# validate this.

# We perform the ADF and KPSS tests on the second differenced log transformed data.

adf.test(diff2_log_df_AP, alternative = "stationary")

# Since p-value (0.01) is less than alpha (0.05), we reject the null hypothesis.
# This means that there is no evidence of non-stationarity after second 
# differenced log transformation.

kpss.test(diff2_log_df_AP)

# Since p-value (0.1) is more than alpha (0.05), we fail to reject the 
# null hypothesis. This means that there is no evidence to show that the 
# transformed time series is non-stationary. This result is consistent with the 
# ADF test result and the seasonality no longer dominates the time series.

# The stationarity has been achieved and hence, we can build ARIMA models.

# We plot the correlograms for analyzing the transformed time series.
# This can be done using the acf () and pacf () functions.

acf(diff2_log_df_AP)

# From the ACF plot of the stationary time series having neither trend nor 
# stationarity, we can see that cut off takes places after the first lag.
# This means d = 1. There is a positive peak and negative peak after which
# there is a decreased sinusoidal pattern in the peaks. Hence, we can say p = 2.

pacf(diff2_log_df_AP)

# From the PACF plot of the stationary time series having neither trend nor 
# stationarity, we can see that cut off takes places after the first positive lag.
# Hence, q = 1. 

acf(diff_log_df_AP)

# From the ACF plot of the stationary time series having no trend but 
# stationarity, we can see that cut off takes places after the first lag.
# This means D = 1. There is a sinusoidal pattern which means there is no AR
# process. Hence, P = 0.

pacf(diff_log_df_AP)

# From the PACF plot of the stationary time series having no trend but 
# stationarity, we can see that there is no cutoff. Hence, Q = 0.

# Creating 3 models using SARIMA.

arima_model1 <- sarima(df_AirPassengers,2,1,1,0,0,0,12)

arima_model2 <- sarima(df_AirPassengers,0,0,0,0,1,0,12)

# We put q = 0 to compare the model performance with the ideal model (q = 1).

arima_model3 <- sarima(df_AirPassengers,2,1,0,0,1,0,12)

# Plotting the residuals of the models.

par(mfrow = c(1,3))

plot(arima_model1$fit$residuals,main = 'Residuals of model 1',ylab = 'Residuals')

plot(arima_model2$fit$residuals,main = 'Residuals of model 2',ylab = 'Residuals')

plot(arima_model3$fit$residuals,main = 'Residuals of model 3',ylab = 'Residuals')

par(mfrow = c(1,1))

# Checking the normality of the residuals.

checkresiduals(arima_model1$fit$residuals)

checkresiduals(arima_model2$fit$residuals)

checkresiduals(arima_model3$fit$residuals)

# Performing the Ljung-Box test on the model residuals.

# H0: The residuals of the time series are independent.
# HA: The residuals of the time series are correlated.

Box.test(arima_model1$fit$residuals,type = "Ljung-Box")

Box.test(arima_model2$fit$residuals,type = "Ljung-Box")

Box.test(arima_model3$fit$residuals,type = "Ljung-Box")

# The p-value (2.2e-16) for the second model is less than alpha (0.05). 
# This means we reject the null hypothesis and can conclude that the residuals 
# are correlated in the second model. In the other models, p-value is significant.
# Hence, we can say that there is no evidence of dependence in the residuals.

# We need to find the correlation between the data and the residuals. 
# This can be done by applying the acf () function on the random components. 
# Since there could be negative values, we find the sum of squares of the acf 
# from both the models. The model type with the least value is the more appropriate .

sum(acf(arima_model1$fit$residuals,na.action = na.omit)$acf^2)

sum(acf(arima_model2$fit$residuals,na.action = na.omit)$acf^2)

sum(acf(arima_model3$fit$residuals,na.action = na.omit)$acf^2)

# We can see that the model with least Sum of Squares is the third model.
# We can also verify this by checking the AIC values. The model with the lowest
# AIC is the best model.

print(arima_model1$fit$aic)

print(arima_model2$fit$aic)

print(arima_model3$fit$aic)

# We can see that the 3rd model has the lowest AIC among all. 
# Hence, it is the best model.

auto_arima_model <- auto.arima(df_AirPassengers)

# Auto ARIMA would be a good option for this dataset since it avoids the manual
# calculation of the orders thereby reducing human error. 
# Model 3 would also work for prediction. 

# Question:- 

# • What model do you consider is better for this dataset, the Classical or the 
# ARIMA model? Support your answer.

# Answer:-

# We can compare the classical model with the ARIMA model using AIC as GOF.

hw_mul_AP$model$aic

arima_model3$fit$aic

auto_arima_model$aic

# The AIC of both the ARIMA models is less than AIC of classical model (Holt-WInter's).
# Therefore, the ARIMA model is better for this dataset.

# Forecast

# Make a forecast for the next 5 periods for both cases, Classical and ARIMA models.

# Question:- 

# • In the classical model, carry out the prediction for multiplicative and 
# additive scenarios and compare the results.

# Answer:-

# Forecasting for the next 5 periods using the classical models.

fore_hw_mul <- forecast(hw(df_AirPassengers, seasonal = "multiplicative"
                           ,h = 5 * frequency(df_AirPassengers)))

par(mfrow = c(1,2))

plot(fore_hw_mul,xlab = 'Time',ylab = "Number of passengers (1000's)")

fore_hw_add <- forecast(hw(df_AirPassengers, seasonal = "additive"
                           ,h = 5 * frequency(df_AirPassengers)))

plot(fore_hw_add,xlab = 'Time',ylab = "Number of passengers (1000's)")

par(mfrow = c(1,1))

# The forecast of the multiplicative model appears better than the forecast of 
# the additive model.

# Question:- 

# • For the ARIMA models, carry out the prediction using the auto.arima models 
# recommended by R and the model you chose during your analysis. Compare the results.

# Answer:-

# Forecasting for the next 5 periods using the ARIMA models.

fore_auto_arima <- forecast(auto_arima_model,h = 5 * frequency(df_AirPassengers))

par(mfrow = c(1,2))

plot(fore_auto_arima,xlab = 'Time',ylab = "Number of passengers (1000's)")

fore_arima_model3 <- sarima.for(df_AirPassengers,2,1,0,0,1,0,12,plot = F
                                ,n.ahead = 5 * frequency(df_AirPassengers)
                                ,plot.all = F)

ts.plot(df_AirPassengers,fore_arima_model3$pred
        ,ylab = "Number of passengers (1000's)"
        ,main = 'Forecasts from ARIMA(2,1,0)(0,1,0)[12]'
        ,lty = c(1,10))

par(mfrow = c(1,1))

# The forecast of the auto.arima model (ARIMA(2,1,1)(0,1,0)[12]) appears better
# than the forecast of the model chosen from analysis.

# Question:- 

# • Make a comparison of the results in the predictions using the Classical and 
# the ARIMA models.

# Answer:-

# Comparing the forecasts results from the ARIMA and classical models.

par(mfrow = c(1,2))

plot(fore_hw_mul,xlab = 'Time',ylab = "Number of passengers (1000's)")

plot(fore_hw_add,xlab = 'Time',ylab = "Number of passengers (1000's)")

plot(fore_auto_arima,xlab = 'Time',ylab = "Number of passengers (1000's)")

ts.plot(df_AirPassengers,fore_arima_model3$pred
        ,ylab = "Number of passengers (1000's)"
        ,main = 'Forecasts from ARIMA(2,1,0)(0,1,0)[12]'
        ,lty = c(1,10))

par(mfrow = c(1,1))

# From the plots we can say that the ARIMA models yields better results than the 
# classical models. 

# The best model is the auto.arima model (ARIMA(2,1,1)(0,1,0)[12]).

# Question:- 

# • Check the power of prediction of your models by forecasting the last periods
# of your time series and comparing them with the actual observed values for 
# such period you have predicted.

# Answer:-

# We can check the accuracy by forecasting the last periods of the time series
# and comparing them with the actual values.

# Storing the observations except for the year end data.

AP_prev <- window(df_AirPassengers,start = start(df_AirPassengers)[1]
                  ,end = c(end(df_AirPassengers)[1]-1,frequency(df_AirPassengers))
                  ,frequency = frequency(df_AirPassengers))

# Forecasting the next year's data to compare with Actual Time Series Data.

fore_hw_mul_prev <- forecast(hw(AP_prev, seasonal = "multiplicative"
                                ,h = frequency(AP_prev)))

fore_hw_add_prev <- forecast(hw(AP_prev, seasonal = "additive"
                                ,h =  frequency(AP_prev)))

# We need to rebuild the auto.arima model since we are taking a subset of the
# original data.

auto_arima_model_prev <- auto.arima(AP_prev)

print(auto_arima_model_prev)

# The new ARIMA model is ARIMA(1,1,0)(0,1,0)[12].

fore_auto_arima_prev <- forecast(auto_arima_model_prev,h = frequency(AP_prev))

fore_arima_model3_prev <- sarima.for(AP_prev,2,1,0,0,1,0,12,plot = F
                                     ,plot.all = F,n.ahead = frequency(AP_prev))

# Plotting forecasts of classical models.

par(mfrow = c(1,3))

plot(fore_hw_mul_prev,xlab = 'Time',ylab = "Number of passengers (1000's)")

plot(fore_hw_add_prev,xlab = 'Time',ylab = "Number of passengers (1000's)")

plot(df_AirPassengers,ylab = "Number of passengers (1000's)"
     ,main = 'Original Time Series Data')

# Plotting forecasts of ARIMA models.

plot(fore_auto_arima_prev,xlab = 'Time',ylab = "Number of passengers (1000's)")

ts.plot(AP_prev,fore_arima_model3_prev$pred,ylab = "Number of passengers (1000's)"
        ,main = 'Forecasts from ARIMA(2,1,0)(0,1,0)[12]',lty = c(1,10))

plot(df_AirPassengers,ylab = "Number of passengers (1000's)"
     ,main = 'Original Time Series Data')

par(mfrow = c(1,1))

# From the plots we can say that the forecasts from ARIMA model are better than 
# the classical models. We can use ARIMA(2,1,0)(0,1,0)[12] or ARIMA(2,1,1)(0,1,0)[12]
# for prediction.

# 2. PCA and FACTOR ANALYSIS

# Preliminary analysis

# Question:- 

# Open the SPSS file into R or RStudio and provide a preliminary description 
# of the dataset using a descriptive summary.

# Answer:-

# getwd() returns the current working directory of the project, where we have 
# saved the data for the assignment.

# Creating file paths for the input dataset.

file_path <- paste(getwd(),'/PCA_STAT9005 Project.sav',sep = '')

car_data <- read.spss(file_path, to.data.frame = T)

dim(car_data)

# The dataset contains 93 observations and 16 features.

# We now view the descriptive summary of the Car Model Dataset.

view(dfSummary(car_data, graph.magnif = 0.85, na.col = F,varnumbers = F
               , valid.col = F))

# The histograms of numeric variables Price, EngineSize, Fuel.tank.capacity, 
# MPG.city, Horsepower and Length are skewed to the right, while the other
# numeric values are skewed to the left. We also need to remove some columns 
# that are not essential for the analysis.

# Checking missing values in the car model dataset.

sum(is.na(car_data))

# There are no missing values in the dataset.

# Storing the numeric column names in a variable.

numeric_columns <- colnames(car_data[,unlist(lapply(car_data,is.numeric))])

# We generate boxplots to check for outliers in the data.

par(mfrow = c(3,3))

for (colname in numeric_columns){
  
  # Checking if the column contains integer or float values.
  
  if (is.integer(car_data[,colname]) || is.numeric(car_data[,colname])){
    
    # Generating the boxplot.
    
    boxplot(x = car_data[,colname],xlab = colname
            ,main = paste('Boxplot of ',colname,sep = ''))
    }
}

par(mfrow = c(1,1))

# From the boxplots we can see that there are very less outliers in the data.

# Question:- 

# Explain the commands used in R and the manipulation performed to prepare the 
# data for a data reduction analysis.

# Answer:-

# For performing Principal Component Analysis (PCA) and Factor Analysis (FA),
# we need to ensure that majority of the variables are numerical.
# Hence, we need to manipulate the data so that we have minimal categorical values.

# The 57th observation in the "Cylinders" variable is inconsistent with the 
# other observations. It holds the value "rotary" while the other observations
# hold numeric values. Hence, the 57th observation is to be removed.

# We only need the categorical information provided by "Manufacturer" and "Model".

car_data_origin <- car_data[-57,-c(13,16)]

car_data_cat <- car_data[-57,-c(1,3,13,16)]

# We need to assign the row names as the "Manufacturer" and "Model" for simplicity.

Manufacturer <- NULL

Model <- NULL

var_Man_Mod <- NULL

for (i in 1:nrow(car_data_origin)){
  
  Manufacturer <- rbind(Manufacturer, car_data_cat[i,"Manufacturer"])
  
  Model <- rbind(Model, car_data_cat[i,"Model"])
  
  var_Man_Mod <- rbind(var_Man_Mod, paste(car_data_cat[i,"Manufacturer"]
                                          ,car_data_cat[i,"Model"]))
  
}

# Create a copy of the car model dataset and assign the row names.

car_origin_df <- data.frame(cbind(car_data_origin[,c(1,3)],car_data_cat[,1:10])
                            ,row.names = var_Man_Mod)

# Creating a dataset with the numeric values only.

car_df_num <- data.frame(car_data_cat[,1:10], row.names = var_Man_Mod)  

# We need to transform the "Cylinders" data into numeric data.

car_df_num[,'Cylinders'] <- as.numeric(car_df_num[,'Cylinders'])

car_origin_df[,'Cylinders'] <- car_df_num[,'Cylinders']

# The numeric data for performing PCA and FA is ready.

# Question:- 

# Study the correlation between the variables of the dataset using the 
# correlation matrix (e.g. with the command “pairs”). Plot the relationship of 
# 3 pairs of variables of your choice and fit a regression line.

# Answer:-

# The pairs command will generate the scatterplots between pairs of variables
# of a dataframe. We will create two functions where we modify the pairs plot
# such that the upper triangle will hold the correlation coefficients, the lower
# triangle will hold the scatterplots and the diagonal elements would hold the
# histograms of each numeric variable.

# Creating a function that would plot the histograms in the diagonal.

hist_panel <- function(x){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5) )
  hist_plot <- hist(x, plot = FALSE)
  breaks <- hist_plot$breaks; nB <- length(breaks)
  y <- hist_plot$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = "red")
}

# Creating a function that would plot the correlations on the upper panels.

corr_panel <- function(x, y, digits = 2, prefix = "", cex.cor){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- (cor(x, y))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 2
  text(0.5, 0.5, txt, cex = cex.cor)
}

# Creating the pairwise plot with the diagonal elements holding the univariate
# histograms and the panels holding the correlation coefficients.

pairs(car_df_num, diag.panel = hist_panel, upper.panel = corr_panel)

# We can see that there are strong correlations between pairs of variables
# ("Fuel.tank.capacity", "Weight"), ("Price", "MPG.city") and 
# ("Price", "Horsepower"). We can visualize this by fitting the regression 
# lines in the scatterplots.

plot(x = car_df_num[,'Fuel.tank.capacity'],y = car_df_num[,'Weight'],ylab = 'Weight'
     ,xlab = 'Fuel tank capacity',main = 'Plot of Weight vs Fuel Tank Capacity')

abline(lm(Weight ~ Fuel.tank.capacity,car_df_num),col = 'red')

plot(x = car_df_num[,'Price'],y = car_df_num[,'MPG.city'],ylab = 'MPG city'
     ,xlab = 'Price',main = 'Plot of MPG city vs Price')

abline(lm(MPG.city ~ Price,car_df_num),col = 'blue')

plot(x = car_df_num[,'Price'],y = car_df_num[,'Horsepower'],ylab = 'Horsepower'
     ,xlab = 'Price',main = 'Plot of Horsepower vs Price')

abline(lm(Horsepower ~ Price,car_df_num),col = 'green')

# Question:- 

# Generate four 3D plots to represent the correlation between the variables 
# “Cylinders”, “EngineSize”, “MPG.city”, “Horsepower” and “Fuel.tank.capacity”. 
# Combine the variables in an appropriate manner to plot a 2D-regression plane 
# on each of the 3D plots for the following relations:
# Cylinders on EngineSize and MPG.city
# Cylinders on EngineSize and Horsepower
# Fule.tank.capacity on Horsepower and MPG.city
# Fuel.tank.capacity on Cylinders and MPG.city

# Based on the relationship between the variables, what would be (if any) your 
# initials guess regarding the latent variables underlying in the dataset?

# Answer:-

# For generating 3D plots we need to invoke the "scatterplot3d" library.

library("scatterplot3d")

attach(car_df_num)

# Generating 3D plot for Cylinders on EngineSize and MPG.city.

plot3D <- scatterplot3d(EngineSize,MPG.city,Cylinders,type = "p"
                        ,highlight.3d = T, pch = 20)

# Generating the regression line for Cylinders on EngineSize and MPG.city.

reg_line <- lm(Cylinders ~ EngineSize + MPG.city, data = car_df_num)

plot3D$plane3d(reg_line,draw_polygon = T, draw_lines = T)

# Generating 3D plot for Cylinders on EngineSize and Horsepower.

plot3D2 <- scatterplot3d(EngineSize,Horsepower,Cylinders,type = "p"
                         ,highlight.3d = T, pch = 20)

# Generating the regression line for Cylinders on EngineSize and Horsepower.

reg_line2 <- lm(Cylinders ~ EngineSize + Horsepower, data = car_df_num)

plot3D2$plane3d(reg_line2,draw_polygon = T, draw_lines = T)

# Generating 3D plot for Fuel.tank.capacity on Horsepower and MPG.city.

plot3D3 <- scatterplot3d(Horsepower,MPG.city,Fuel.tank.capacity,type = "p"
                         ,highlight.3d = T, pch = 20)

# Generating the regression line for Fuel.tank.capacity on Horsepower and MPG.city.

reg_line3 <- lm(Fuel.tank.capacity ~ Horsepower + MPG.city, data = car_df_num)

plot3D3$plane3d(reg_line3,draw_polygon = T, draw_lines = T)

# Generating 3D plot for Fuel.tank.capacity on Cylinders and MPG.city

plot3D4 <- scatterplot3d(Cylinders,MPG.city,Fuel.tank.capacity,type = "p"
                         ,highlight.3d = T, pch = 20)

# Generating the regression line for Fuel.tank.capacity on Cylinders and MPG.city.

reg_line4 <- lm(Fuel.tank.capacity ~ Cylinders + MPG.city, data = car_df_num)

plot3D4$plane3d(reg_line4,draw_polygon = T, draw_lines = T)

detach(car_df_num)

# From the plots we can say that the relationships between the variables are
# significant as the regression line covers most of the points.
# We can say that the latent variables are EngineSize, MPG.city and Horsepower.

# Question:- 

# Carry out an appropriate test to check whether the dataset is suitable for 
# implementing data reduction techniques (e.g. using Barlett test).
# Comment on your results.

# Answer:-

# We can perform Bartlett's test of sphericity to check if we can perform
# data reduction techniques. We need to invoke the 'parameters' library to 
# perform this test.

library(parameters)

# Bartlett's test of sphericity:
# H0 = Variables are uncorrelated i.e. correlation matrix is an identity matrix.
# HA = Variables are correlated and suitable for factor analysis i.e. 
# correlation matrix is not an identity matrix.

check_sphericity(car_df_num) 

# Since p-value is less than alpha (0.05), we reject the Null Hypothesis.
# There is sufficient significant correlation in the data for factor analysis 
# (Chisq(45) = 1101.15, p < .001).

# Principal Components

# Question:- 

# Perform PCA analysis to the dataset using 3 different commands and libraries in R. 
# For each model, identify and compare the results of:
# a. Eigenvalues and eigenvectors for each component.

# Answer:-

# PCA can be done in R using 3 commands namely prcomp, princomp, and PCA.
# We need to invoke the 'FactoMineR' library for using PCA command.

library(FactoMineR)

pca1 <- prcomp(car_df_num, scale. = T)

pca2 <- PCA(car_df_num, graph = F)

pca3 <- princomp(car_df_num, cor = T)

# Comparing the eigenvalues of the 3 commands.

cat('\nEigenvalues of prcomp command:',(pca1$sdev)^2)

cat('\nEigenvalues of PCA command:',pca2$eig[,'eigenvalue'])

cat('\nEigenvalues of princomp command:',(pca3$sdev)^2)

# The eigenvalues of the 3 commands are same. If the eigenvalues are same, then
# the eigenvectors would be same as well.

# Question:- 

# b. Amount of information explained by each component as well as the cumulative
# variance.

# Answer:- 

# We can view the amount of information by each PCA component using the summary.

# Amount of information and cumulative variance from result of prcomp command.

summary(pca1)

# Amount of information and cumulative variance from result of PCA command.

pca2$eig[,c('percentage of variance','cumulative percentage of variance')]

# Amount of information and cumulative variance from result of princomp command.

summary(pca3)

# We can see that the results from the 3 commands are identical.

# Question:- 

# c. Identify the resulting factor loadings and scores in the analysis. 
# Explain the way these elements are computed in PCA.

# Answer:- 

# Factor loadings and scores from result of prcomp command.

head(pca1$rotation,5)

head(pca1$x,5)

# Factor loadings and scores cannot be computed from result of PCA command.

# Factor loadings and scores from result of princomp command.

head(pca3$loadings,5)

head(pca3$scores,5)

# The factor loading results of prcomp and princomp commands are identical.
# The PCA scores are almost identical between the results of the commands. 
# Factor Loadings are calculated using the eigenvalues and the slop co-ordinates.
# Factor Loading = u(sqrt(lambda))/sqrt(var(X)*lambda).
# PCA scores are calculated by multiplying the input matrix and the matrix
# formed using the eigenvectors of the correlation matrix that is derived from 
# the input matrix.

# Question:- 

# d. Study the correlation between the original variables and each principal 
# component. What variables are represented by the first two principal components? 
# Use factor matrix and biplot to comment and support your findings.

# Answer:- 

# The correlation between the original variables and principal components can
# be studied by viewing at the factor loadings. Since the factor loading
# results of prcomp and princomp commands are identical, we will be using the
# result of prcomp command.

pca1$rotation

# The correlations are very weak or almost moderate between the original 
# variables and the principal components. Only few of the correlations are strong.

# We use biplot to understand the variables represented by PC1 and PC2.

biplot(pca1,scale = 0,cex = 0.7)

# The variables Cylinders, EngineSize, Fuel.tank.capacity, Weight and 
# Length are more related to PC1 while the rest are more related to PC2.

# Question:- 

# e. Use a screeplot to decide how many components to retain. 
# Explain why you are choosing these components. Is this decision supported by 
# what have you observed in the previous question using the factor loading and 
# biplots?

# Answer:-

# Generating the screeplot.

screeplot(pca1, type = "line", main = "Screeplot of PCA result")

# From the screeplot we can say that it is best to retain 3 components.
# After the second component, the graph almost saturates. This is because 
# most of the information is explained by the first two principal components.
# This decision is supported by the factor loadings and biplot.

# Question:-

# f. Use the PC scores to interpret the relation between the observations and 
# each of the components. Link the values of the scores on the matrix with the
# coordinates of the observations in regards to the PC’s (use the biplot for 
# coordinates)

# Answer:-

# Since the PC scores from results of prcomp and princomp commands are almost 
# identical, we will be using the result of prcomp command.

biplot(pca1$x,pca1$rotation)

# Most of the observations are pointed towards PC1 than PC2. This is due to the
# strong correlation between PC1 and observations.

# Question:-

# g. Use the libraries “FactoMineR” and “factoextra”, and the commands 
# “fviz_eig”, “fviz_pca_var”, “fviz_pca_ind” from the libraries to represent:
# - A screeplot and correlation circle of the dataset.
# - A biplot representing the cars by “Origin” – USA and NonUSA.
# Do you observe any difference in the biplot between the USA manufacturers and 
# the rest of carmakers? Which variables do you think are involved in 
# these differences?

# Answer:- 

# Invoking the libraries “FactoMineR” and “factoextra”.

library(FactoMineR)

library(factoextra)

# Generating screeplot and correlation circle.

car_df_num_pca <- PCA(car_df_num, graph = T)

fviz_eig(car_df_num_pca, addlabels = TRUE)

# Generating the biplot representing the cars by “Origin” – USA and NonUSA. 

fviz_pca_ind(car_df_num_pca,geom.ind = "point",addEllipses = T,
             col.ind = car_origin_df[,'Origin'],palette = c("blue", "red")
             ,legend.title = "Origin of Manufacture")

fviz_pca_var(car_df_num_pca, col.var = "red")

# There is a slight difference in the biplots between the USA manufacturers and 
# the rest of carmakers. For the cars manufactured in USA, the variables 
# MPG.city, Price and Horsepower are the significant components.
# For the other manufacturers, the variables Cylinders, EngineSize, Weight, 
# Fuel.tank.capacity and Length are the significant components.

# Factor Analysis

# Question:-

# Carry out a Factor analysis on the dataset using the “factanal” from the 
# library “psych”. Perform the analysis considering 1, 2 and 3 underlying 
# variables and comment on the results.

# Answer:-

# Invoking the library psych.

library(psych)

# Performing the factor analysis.

fa1 <- factanal(car_df_num, 1, rotation = "varimax")

fa2 <- factanal(car_df_num, 2, rotation = "varimax")

fa3 <- factanal(car_df_num, 3, rotation = "varimax")

plot(fa1$loadings,main = 'Plot of FA loadings with number of factors = 1')

text(fa1$loadings,labels = names(car_df_num),cex = 0.75)

plot(fa2$loadings,main = 'Plot of FA loadings with number of factors = 2')

text(fa2$loadings,labels = names(car_df_num),cex = 0.75)

plot(fa3$loadings,main = 'Plot of FA loadings with number of factors = 3')

text(fa3$loadings,labels = names(car_df_num),cex = 0.75)

# When number of factors = 1, most of the weights are higher than 0.3.
# When number of factors = 2, only some of the weights are higher than 0.3.
# When number of factors = 3, most of the weights are lesser than 0.3.
# The weights are to be checked when the values are higher than 0.3.

print(fa1, digits = 3, cutoff = .3, sort = T)

# Around 57.1% of the variance is explained by factor 1 when number of factors = 1.

print(fa2, digits = 3, cutoff = .3, sort = T)

# Around 39.1% of the variance is explained by factor 1 and 32% of the variance 
# is explained by factor 2 when number of factors = 2.

print(fa3, digits = 3, cutoff = .3, sort = T)

# Around 34.1% of the variance is explained by factor 1, 23% of the variance 
# is explained by factor 2 and 21% of the variance is explained by factor 3
# when number of factors = 3.
# As the number of factors increases the variability explained by the first
# factor decreases.

# Question:-

# Use the library “nFactors” to determine the number of factors to extract. 
# Is this consistent with what you obtained from previous PCA steps?
# Interpret and comment on the factor loadings you have obtained.

# Answer:-

# Invoking the library nFactors.

library(nFactors)

# We use the fa.parallel command to identify the number of factors required.

fa.parallel(car_df_num,fm = "minres",fa = 'fa')

# We can see that we need only 2 factors and no additional components.
# This is consistent with the PCA results.

fa4 <- fa(car_df_num,nfactors = 2, rotate = 'varimax', fm = 'minres')

# Comparing the factor loadings.

fa4$loadings

# The loadings are almost similar to the factanal results. Majority of the 
# loadings are above 0.3 and need to be re-investigated.
# Here, around 43% of the variance is explained by factor 1 and 30% of the  
# variance is explained by factor 2. This is a better result than factanal command.

# Question:-

# Use “fa” command from library “FactoMineR” and compare the result with your 
# previous one. Use different methods for factor extraction (e.g. Maximum 
# likelihood and Principal axis) and compare the results. Do you find any 
# difference in the factors obtained using different methods?

# Answer:-

# Using Maximum Likelihood for feature extraction.

fa5 <- fa(car_df_num,nfactors = 2, rotate = 'varimax', fm = 'ml')

# Using Principal axis for feature extraction.

fa6 <- fa(car_df_num,nfactors = 2, rotate = 'varimax', fm = 'pa')

print(fa5)

print(fa6)

# Plotting the different FA models.

par(mfrow = c(1,2))

plot(fa5,main = 'Factor analysis (Maximum Likelihood)')

plot(fa6,main = 'Factor analysis (Principal axis)')

biplot(fa5,main = 'Factor analysis (Maximum Likelihood)')

biplot(fa6,main = 'Factor analysis (Principal axis)')

par(mfrow = c(1,1))

# The plots are not much different between the two models. 

# Checking the loadings of the FA models.

fa5$loadings

fa6$loadings

# There is a minor difference in the factor loadings when different methods are
# used. 

# Question:-

# Use the command “fa.diagram” to obtain a path diagram of the items’ 
# factor loadings. Build two diagrams using different number of factors. 
# Interpret and compare the results for the different number of factors chosen.

# Answer:-

# Building two models using different number of factors.

fa7 <- fa(car_df_num,nfactors = 2, rotate = 'varimax', fm = 'ml')

# Using Principal axis for feature extraction.

fa8 <- fa(car_df_num,nfactors = 3, rotate = 'varimax', fm = 'ml')

# Creating path diagrams for the different factor models.

par(mfrow = c(1,2))

fa.diagram(fa7,main = 'Factor Analysis (Maximum Likelihood) with 2 factors')

fa.diagram(fa8,main = 'Factor Analysis (Maximum Likelihood) with 3 factors')

par(mfrow = c(1,1))

# As the number of factors increases, the weights of each variable decrease
# slightly. There is no correlation in the factors, and we can see that this is
# akin to PCA.

# Question:-

# Models

# Write the final PCA and FA models for this dataset.

# Answer:-

# The prcomp command would be used as final PCA model.
# The fa command would be used as final FA model.

# Second analysis

# Question:-

# Split the dataset into two groups by its origin (USA-cars and NonUSA-cars).

# Repeat the PCA and FA analysis (using one command will be enough here).

# Do you find any difference in the results of the two different groups?

# Do you find any differences with the results obtained previously for the for 
# dataset?

# Answer:-

# Invoking dplyr library for data manipulations.

library("dplyr")

# Creating separate datasets for holding different origins.

car_origin_df_USA <- car_origin_df %>% filter(Origin == 'USA')

car_df_num_USA <- car_origin_df_USA[,colnames(car_df_num)]

car_origin_df_non_USA <- car_origin_df %>% filter(Origin != 'USA')

car_df_num_non_USA <- car_origin_df_non_USA[,colnames(car_df_num)]

# Performing PCA.

pca_USA <- prcomp(car_df_num_USA, scale. = T)

pca_non_USA <- prcomp(car_df_num_non_USA, scale. = T)

# Comparing EigenValues.

(pca_USA$sdev)^2

(pca_non_USA$sdev)^2

# The eigenvalues are different for the different Origins. This applies for the
# eigenvectors as well. 

# Amount of information and cumulative variance.

summary(pca_USA)

summary(pca_non_USA)

# The amount of information varies between the different origins.

# Factor loadings and scores.

head(pca_USA$rotation,5)

head(pca_non_USA$rotation,5)

head(pca_USA$x,5)

head(pca_non_USA$x,5)

# Most of the factor loadings and PCA scores of USA data are varying significantly 
# from the values of Non-USA data.

# Comparing the biplots.

biplot(pca_USA,scale = 0,cex = 0.7, main = "Biplot of US data PCA")

biplot(pca_non_USA,scale = 0,cex = 0.7, main = "Biplot of Non US data PCA")

# In both the biplots, the variables Passengers and RPM are more related to PC2 
# while the rest are more related to PC1.

# Generating the screeplots.

par(mfrow = c(1,2))

screeplot(pca_USA, type = "line", main = "Screeplot of US data PCA")

screeplot(pca_non_USA, type = "line", main = "Screeplot of Non US data PCA")

par(mfrow = c(1,1))

# From the screeplots we can say that it is best to retain 3 components.
# After the second component, the graph almost saturates. 

# Biplots for observations.

par(mfrow = c(1,2))

biplot(pca_USA$x,pca_USA$rotation, main = "Observational biplot of US data PCA")

biplot(pca_non_USA$x,pca_non_USA$rotation
       , main = "Observational biplot of Non US data PCA")

par(mfrow = c(1,1))

# In both the biplots, most of the observations are pointed towards PC1 than PC2. 
# This is due to the strong correlation between PC1 and observations.

# Building two models for USA and Non USA Data.

fa_USA <- fa(car_df_num_USA,nfactors = 2, rotate = 'varimax', fm = 'ml')

# Using Principal axis for feature extraction.

fa_non_USA <- fa(car_df_num_non_USA,nfactors = 2, rotate = 'varimax', fm = 'ml')

# Creating path diagrams for the different factor models.

par(mfrow = c(1,2))

fa.diagram(fa_USA,main = 'Factor Analysis (USA) with 2 factors')

fa.diagram(fa_non_USA,main = 'Factor Analysis (Non USA) with 3 factors')

par(mfrow = c(1,1))

# The path diagrams are similar between the USA and non USA data.

# Creating plots for the different models.

par(mfrow = c(1,2))

plot(fa_USA,main = 'Factor analysis (USA)')

plot(fa_non_USA,main = 'Factor analysis (Non USA)')

biplot(fa_USA,main = 'Factor analysis (USA)')

biplot(fa_non_USA,main = 'Factor analysis (Non USA)')

par(mfrow = c(1,1))

# The plots are similar between the USA and non USA data.

# Checking the loadings of the FA models.

fa_USA$loadings

fa_non_USA$loadings

# There is a minor difference in the factor loadings due to the difference in data.
# There are minor differences in the results of the two different groups.
# There are major differences with the results obtained previously for the dataset.
