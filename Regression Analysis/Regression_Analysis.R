# Analysis of Tree Dataset

# a) Make a numerical and graphical summary of the data.

# Invoking R libraries.

# install.packages("datarium")

library('asbio')
library('pwr')
library('tidyverse')
library('dplyr')
library('readxl')
library('mice')
library('summarytools')
library('faraway')
library('VIM')
library('caret')
library('datarium')
library("MASS")
library("car")

# getwd() returns the current working directory of the project, where the data 
# is saved for the assignment.

file_path <- paste(getwd(),'/treeG.csv',sep = '')

# Storing the tree dataset in a data frame.

treeG_df_source <- read.csv(file = file_path,stringsAsFactors = 1)

# Checking if the dataset contains missing values.

sum(is.na(treeG_df_source))

# There are no missing values in the dataset.

summary(treeG_df_source)

# Taking backup of the original dataset. 

treeG_df <- treeG_df_source

# Univariate Analysis:-

# Here, univariate plots are created to understand the behaviour of each field.

# Plotting Histograms, Boxplots and Barplots for understanding the distribution
# of data.

par(mfrow = c(length(colnames(treeG_df)),length(colnames(treeG_df))))

for (colname in colnames(treeG_df)){
  
  # Checking if the column contains integer or float values.
  
  if (is.integer(treeG_df[,colname]) || is.numeric(treeG_df[,colname])){
    
    # Generating the histogram and boxplot.
    
    hist(x = treeG_df[,colname],xlab = colname
         ,main = paste('Histogram of ',colname,sep = '')
         )
    
    boxplot(x = treeG_df[,colname],xlab = colname
            ,main = paste('Boxplot of ',colname,sep = '')
            )
    
  }
}

# The histograms of Diam and Vol predictors resemble a right-skewed normal 
# distribution. From the boxplots, we can see that there are three outliers in 
# the Dam variable and seven outliers in the Vol variable.

# Bivariate Analysis:-

# Storing the column names of the tree dataset in a variable.

col_names <- colnames(treeG_df)

col_index <- 1:length(col_names)

par(mfrow = c(1,1))

for (i in col_index){
  for (j in col_index){
    if(i!=j){
      
      cat('Bivariate analysis between',col_names[i],'and',col_names[j],'\n')
      
      # If both the dependent and independent variables are numeric, then a
      # scatter plot is generated.
      
      if(is.numeric(treeG_df[,col_names[i]])){
        if(is.numeric(treeG_df[,col_names[j]])){
          plot(x = treeG_df[,col_names[j]]
               ,y = treeG_df[,col_names[i]]
               ,xlab = col_names[j]
               ,ylab = col_names[i]
               ,main = paste('Scatterplot of',col_names[i],'vs',col_names[j])
               )
        }
      }
      
      # If one of the variables is numeric, and the other categorical, then a 
      # box plot is generated.
      
      if(is.numeric(treeG_df[,col_names[i]])){
        if(is.factor(treeG_df[,col_names[j]])){
          plot(x = treeG_df[,col_names[j]]
               ,y = treeG_df[,col_names[i]]
               ,xlab = col_names[j]
               ,ylab = col_names[i]
               ,main = paste('Boxplot of',col_names[i],'vs',col_names[j])
               )
        }
      }
      
      if(is.numeric(treeG_df[,col_names[j]])){
        if(is.factor(treeG_df[,col_names[i]])){
          plot(x = treeG_df[,col_names[i]]
               ,y = treeG_df[,col_names[j]]
               ,xlab = col_names[i]
               ,ylab = col_names[j]
               ,main = paste('Boxplot of',col_names[j],'vs',col_names[i])
          )
        }
      }
      
    }
  }
  col_index <- col_index[-1]
}

# From the scatterplot we can see that there is a positive relationship between
# Diam and Vol.
# Storing the numeric columns in a vector.

numeric_columns <- colnames(treeG_df[,unlist(lapply(treeG_df, is.numeric))])

# Checking correlation of numeric variables in Tree Dataset.

cor(treeG_df[,numeric_columns])

# There appears to be a strong relationship between the variables Diam and Vol. 
# Hence, we can fit a linear model.

# b) Fit a model of the form 〖y=β〗_0+β_1 x+e and interpret the value of β_1.

# From the results discussed in (a) we create a linear model.

treeG_model <- lm(Diam ~ Vol,data = treeG_df)

summary(treeG_model)

par(mfrow = c(2,2)) 

plot(treeG_model)

par(mfrow = c(1,1))

# The model equation is Diam = 0.20558 + 0.09648*Vol + e, 'e' is the residual.

# c) Calculate a 95% confidence interval for the β̂_1 coefficient.

SSxx <- sum((treeG_df$Vol - mean(treeG_df$Vol))^2)

SSyy <- sum((treeG_df$Diam - mean(treeG_df$Diam))^2)

SSxy <- sum((treeG_df$Diam - mean(treeG_df$Diam))*(treeG_df$Vol - mean(treeG_df$Vol)))

SSREG <- SSxy^2 / SSxx

RSS <- SSyy - SSREG

R = (SSREG/SSyy)^0.5

Î²1_hat <- SSxy/SSxx

Î²0_hat <- mean(treeG_df$Diam) - Î²1_hat*mean(treeG_df$Vol)

sigma_hat <- (RSS/(nrow(treeG_df) - 2))^0.5

# The confidence intervals on the slope Î²1 is: 
# [Î²Ì1 - t (alpha/2,n-2) * S.E.(Î²Ì1), Î²Ì1 + t (alpha/2,n-2) * S.E.(Î²Ì1)]

# S.E.(Î²Ì1) = sigma_hat/(SSxx)^0.5

SE_beta_1_hat <- sigma_hat/((SSxx)^0.5)

confidence_level <- 0.95

alpha <- 1 - confidence_level

# Calculating the t value using qt() function.

t_val <- abs(qt(alpha/2, nrow(treeG_df)-2))

# Calculating the confidence intervals.

confidence_intervals <- c(Î²1_hat - t_val * SE_beta_1_hat
                          , Î²1_hat + t_val * SE_beta_1_hat
                          )

confint(treeG_model)

# d) Test the hypothesis for the regression model.

# We can see the summary of the model to get the p-values.

summary(treeG_model)

# Since p-value (2.93e-16) is less than alpha (0.05), we reject the null
# hypothesis. Also, the slope (0.20558) is non-zero which implies that the
# variable Vol can be used for modeling the Diam variable.

# We can also test the hypothesis by viewing the ANOVA table.

summary.aov(treeG_model)

# Calculating the f value using qf() function.

f_crit <- qf(alpha,1,nrow(treeG_df) - 2,lower.tail = FALSE)

MSE <- sum(treeG_model$residuals^2)/treeG_model$df.residual

f_stat <- SSREG/MSE

# The F-statistic is 136.1 with (1,53) degrees of freedom and p < 0.001.
# Since, the F-statistic (136.1) is greater than F-critical (4.023017), we 
# conclude that the slope is non-zero and Vol is of value in explaining the 
# variability of the Diam.

# e) Plot the regression line onto a scatterplot of the data and plot a 95% 
# prediction band.

# Answer:- 

# Plotting the regression line on the scatterplot.

plot(x = treeG_df[,'Vol'],y = treeG_df[,'Diam'],xlab = 'Vol',ylab = 'Diam'
     ,main = paste('Scatterplot of Diam vs Vol')
     )

abline(treeG_model,col = 'red')

# Plotting the 95% prediction band.

# Creating the Vol range for prediction.

Vol_range <- seq(min(treeG_df[,'Vol']),max(treeG_df[,'Vol']),by = 0.05)

pred_band <- predict(treeG_model,newdata = data.frame(Vol = Vol_range)
                     ,interval = "prediction",level = 0.95
                     ) 

plot(x = treeG_df[,'Vol'],y = treeG_df[,'Diam'],xlab = 'Vol',ylab = 'Diam'
     ,main = paste('Scatterplot of Diam vs Vol')
     )

abline(treeG_model,col = 'black')

lines(Vol_range, pred_band[,"lwr"], col = "red")

lines(Vol_range, pred_band[,"upr"], col = "red")

# f) Plot the studentized residuals against the fitted values and identify any outliers.

# The studentized residuals can be viewed with the help of plot command.

par(mfrow = c(2,2))

plot(treeG_model)

par(mfrow = c(1,1))

qqPlot(treeG_model)

std_res <- rstandard(treeG_model)

plot(std_res ~ fitted(treeG_model), xlab = "Vol (m^3) (fitted values)"
     ,ylab = "studentized residuals"
     )

abline(0,0,col = 'blue')

abline(h = 2,col = 'red')

# An observation may be considered an outlier if the magnitude of its 
# studentized residual is greater than 2. From the plot we can say that there
# are very few outliers. However, there seems to be no pattern in the variance 
# of the residuals i.e. there is no decrease or increase in the variance with 
# respect to the mean.

# g) Plot the leverage of each case and identify any observations that have high 
# leverage.

# The leverage can be calculated using influence function.

# Formula for leverage (h) = 2 * (k + 1) / n, where 'k' is the number of 
# explanatory variables and 'n' is the number of observations.

# Storing the leverage in a variable.

leverage = 2 * (length(colnames(treeG_df))) / (nrow(treeG_df))

tree_leverage <- lm.influence(treeG_model)$hat

par(mfrow = c(1,1))

plot(tree_leverage, xlab = "Observation", ylab = "Leverage"
     ,main = 'Plot of Leverage vs Observation'
     )

abline(h = leverage,col = 'blue')

# The observations with the highest leverage can be identified using the 
# identify() function.

# identify(tree_leverage,n = 4)

# The 16th, 39th, 43rd and 45th observations have the highest leverages.

# h) Identify the observation that has the largest influence on the estimate of 
# the β̂_1 coefficient.

# The influential points can be determined using Cook's distance from the model 
# diagnostic plot.

par(mfrow = c(1,1))

plot(treeG_model,4)

# The 16th, 27th and 39th observations have the largest Cook's distance which 
# can be considered as the highest influential points.

# Analysis of Divorce Dataset

# a) Make a numerical and graphical summary of the data.

file_path_div <- paste(getwd(),'/divusaF.csv',sep = '')

# Storing the div dataset in a data frame.

div_df_source <- read.csv(file = file_path_div,stringsAsFactors = 1)

# Checking if the dataset contains missing values.

sum(is.na(div_df_source))

# There are no missing values in the dataset.

summary(div_df_source)

# Taking backup of the original dataset. The year attribute is not required
# for analysis and hence it can be dropped from the dataset.

div_df <- div_df_source

div_df[,'year'] <- NULL

# Univariate Analysis:-

# Here, univariate plots are created to understand the behaviour of each field.

# Plotting Histograms, Boxplots and Barplots for understanding the distribution
# of data.

par(mfrow = c(2,2))

for (colname in colnames(div_df)){
  
  # Checking if the column contains integer or float values.
  
  if (is.integer(div_df[,colname]) || is.numeric(div_df[,colname])){
    
    # Generating the histogram and boxplot.
    
    hist(x = div_df[,colname],xlab = colname
         ,main = paste('Histogram of ',colname,sep = '')
         )
    
    boxplot(x = div_df[,colname],xlab = colname
            ,main = paste('Boxplot of ',colname,sep = '')
            )
    
  }
}

par(mfrow = c(1,1))

# The histograms of divorce, unemployed, femlab and military predictors 
# resemble a right-skewed normal distribution whereas the histograms of
# marriage and birth predictors resemble a normal distribution with a minor left
# skew.

# From the boxplots, we can see that there are very few outliers in the 
# unemployed and military variables and only one outlier in the femlab variable.
# The other attributes have no outliers.

# Bivariate Analysis:-

# Pairwise plot would be used for Bi-Variate analysis.

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

# Creating the pairwise plot with the diagonal holding the univariate
# histograms and the panels holding the correlation coefficients.

pairs(div_df, diag.panel = hist_panel, upper.panel = corr_panel)

# We can see that there is a very strong positive relationship between divorce 
# and femlab variables, while there is a very strong negative relationship 
# between variables divorce and marriage. 

# Moderate relationships are found among the variable pairs (divorce, birth),
# (femlab, marriage), (femlab, birth) and (marriage, birth).

# Weak relationships are found among the remainder of the variable pairs.

# b) Fit the model: 〖y=β〗_0+β_1 unemployed+〖β_2 femlab+β〗_3 marriage+
# 〖β_4 birth+β〗_3 military+e 

# The response variable is the divorce variable.

div_model <- lm(divorce ~., data = div_df)

# (i)	Interpret the coefficient for femlab.

summary(div_model)

# The model equation is divorce = 33.793826 + 0.009136*unemployed + 
# 0.284877*femlab -0.281241*marriage -0.101615*birth -0.027808*military + e,
# 'e' is the residual.

# The coefficient for femlab is 0.284877.

# (ii)	Calculate the variance inflation factors for this model and discuss 
# their implications for collinearity in the model.

# The variance inflation could be calculated using the vif() function.

vif(div_model)

# Collinearity can be detected from the variance inflation factor (VIF) 
# associated with each coefficient. If the maximum VIF exceeds 5 then this is 
# an indication of serious collinearity between the explanatory variables.

# Since the VIF is less than five for all the variables, we can say that there 
# exists a very weak collinearity between them.

# (iii)	Create a partial regression plot to examine relationship between birth 
# and divorce adjusted for unemployed, femlab, marriage and military. 

# For creating a partial regression plot, first we need to calculate the 
# residuals for the model birth ~ unemployed, femlab, marriage and military.

div_birth_model <- lm(birth ~.-divorce, data = div_df)

birth_resid <- div_birth_model$residuals

# Next, we need to calculate the residuals for the model divorce ~ unemployed, 
# femlab, marriage and military.

div_divorce_model <- lm(divorce ~.-birth, data = div_df)

divorce_resid <- div_divorce_model$residuals

# We then plot the residuals from the second model (y) against the residuals 
# from the first model (x).

plot(x = birth_resid,y = divorce_resid
     ,xlab = 'birth ~ unemployed, femlab, marriage and military residuals'
     ,ylab = 'divorce ~ unemployed, femlab, marriage and military residuals'
     ,main = paste('Partial regression plot')
     )

# Finally, we can fit a regression line to the two sets of residuals, 
# the slope of the regression line measures the effect of birth on divorce 
# adjusted for unemployed, femlab, marriage and military.

div_birth_divorce_resid_model <- lm(divorce_resid ~ birth_resid)

abline(div_birth_divorce_resid_model, col = 'red')

summary(div_birth_divorce_resid_model)

# We can see a negative relationship between the two sets of residuals which 
# represents the relationship between birth and divorce adjusted for unemployed,
# femlab, marriage and military. 

# The coefficient for birth_resid in the partial regression model is same as the
# coefficient for birth in the full model.

div_model$coefficients['birth']

div_birth_divorce_resid_model$coefficients['birth_resid']

# (iv) Test the hypothesis of the regression model.

# We can see the summary of the model to get the p-values.

summary(div_model)

# From the summary we can see that the F-Statistic is F(5,71) = 74.38 and 
# p-value < 2.2e-16. This F-Statistic compares the intercept-only model 
# (null model) to the fitted model and here we may reject the null hypothesis
# at the 1% confidence level and conclude that at least one of the predictors 
# is associated with divorce.

# (v)	Assess the fit of the model using diagnostic plots, commenting on the 
# assumptions of the regression model and influential points.

# By plotting the model, we can perform the model diagnosis.

# First, we need to check the homoscedasticity of the residuals.

par(mfrow = c(2,2)) 

plot(div_model)

par(mfrow = c(1,1))

# From the Residuals vs Fitted values plot we can see that there is no pattern 
# in the variance of the residuals i.e. there is no decrease or increase in the 
# variance with respect to the mean. The residuals are randomly distributed with
# zero mean and standard deviation (sigma). 

# We then check the studentized residuals for the fitted values.

std_div_res <- rstandard(div_model)

plot(std_div_res ~ fitted(div_model),xlab = "fitted values"
     ,ylab = "studentized residuals"
     )

abline(0,0,col = 'blue')

abline(h = 2,col = 'red')

# An observation may be considered an outlier if the magnitude of its 
# studentized residual is greater than 2. From the plot we can say that there
# are very few outliers. However, there seems to be no pattern in the variance 
# of the residuals i.e. there is no decrease or increase in the variance with 
# respect to the mean.

# We then check the studentized residuals for each predictor in the dataset.

par(mfrow = c(1,1))

plot(std_div_res ~ div_df$unemployed)
plot(std_div_res ~ div_df$femlab)
plot(std_div_res ~ div_df$marriage)
plot(std_div_res ~ div_df$birth)
plot(std_div_res ~ div_df$military)

# In the first plot, the residuals appear to be split into two groups, with one 
# group having a larger variance with the other having a smaller variance. 
# In the second plot there could be one possible outlier in the variable femlab. 
# The residuals in the next couple of plots show no sign of heteroscedasticity. 
# The last plot of the military variable shows three possible outliers. 

# Next, we need to check the normality of the residuals.

par(mfrow = c(1,1))

qqPlot(div_model)

hist(div_model$residuals)

# From the Normal Q-Q plot we can see that the residuals are almost normally
# distributed except for few observations. This can be observed in the histogram
# as well.

# Finally, we need to find the influential points.

# The leverage can be calculated using influence function.

# Formula for leverage (h) = 2 * (k + 1) / n, where 'k' is the number of 
# explanatory variables and 'n' is the number of observations.

# Storing the leverage in a variable.

leverage = 2 * (length(colnames(div_df))) / (nrow(div_df))

tree_leverage <- lm.influence(div_model)$hat

par(mfrow = c(1,1))

plot(tree_leverage, xlab = "Observation", ylab = "Leverage"
     ,main = 'Plot of Leverage vs Observation'
     )

abline(h = leverage,col = 'blue')

# The observations with the highest leverage can be identified using the 
# identify() function.

# identify(tree_leverage,n = 9)

# The 3rd, 7th, 8th, 14th, 24th, 25th, 26th, 37th and 46th observations have 
# the highest leverages.

# This can be determined using Cook's distance from the model diagnostic plot. 

par(mfrow = c(2,2))
plot(div_model)
par(mfrow = c(1,1))

plot(div_model,4)

# The 25th, 26th and 37th observations have the largest Cook's distance which 
# can be considered as the highest influential points.

# c) Use the predict function to calculate the expected divorce rate when 
# unemployed = 8, femlab = 22, marriage = 80 birth =115 and military =2.7

predict(div_model,newdata = data.frame(unemployed = 8
                                       ,femlab = 22
                                       ,marriage = 80
                                       ,birth = 115
                                       ,military = 2.7
                                       )
        ) 

# The expected divorce rate is 5.874062.

# d) Compare the full model to the model where birth and military are excluded 
# using 50 repeats of 10-fold cross validation. Which model would you choose to 
# predict crime rate?

# Setting the seed to control the randomness and reproduce the results.

set.seed(750)

# Using 50 repeats of 10-fold cross validation.

train_cv <- trainControl(method = "repeatedcv",number = 10,repeats = 50)

# Creating the model where birth and military are excluded.

model_excl <- train(divorce ~.-birth-military,data = div_df,method = "lm"
                    ,trControl = train_cv
                    )

# Creating the full model.

model_full <- train(divorce ~.,data = div_df,method = "lm",trControl = train_cv)

# Printing the models.

print(model_excl); print(model_full)

print(model_excl$finalModel); print(model_full$finalModel)

# The RMSE of the excluded model is less than the RMSE of the full model.
# The R square of the excluded model is greater than that of the full model.
# This means that the excluded model is more accurate than the full model.
# However, for choosing the best model we need to use ANOVA() function.

# Test the hypothesis of Anova:

# H0: Î²4=Î²5=0
# HA: Î²4,Î²5 not both equal to 0

anova(model_excl$finalModel,model_full$finalModel)

# From the ANOVA we can see that the F-statistic is F(2, 71) = 5.3194
# and p-value = 0.007036. This F-statistic compares the fit of reduced model to
# the full model at the 1% confidence level.
# In this case, we fail to reject H0 and conclude that there is not
# enough evidence to prove that the variables birth and military are associated 
# with divorce (when they are included in the model).

# Hence, we can use the reduced model for prediction.

# How does varying the number of predictors affect the performance of 
# step-wise regression?

# Setting the seed to control the randomness and reproduce the results.

set.seed(1165)

# Generating the number of explanatory variables randomly.

num_var_vec <- round(runif(100,min = 10,max = 30))

# Generating the number of explanatory variables that are linearly related to Y 
# randomly.

num_lin_var_vec <- round(runif(100,min = 5,max = 15))

# Generating the magnitude of the explanatory variables that are linearly 
# related to Y randomly.

B_mag_vec <- round(runif(100,min = 0.05,max = 0.99),3)

# Storing the number of observations to be generated in a data set.

n_obs = 100 

# Creating two vectors to record the number of variables that were retained 
# incorrectly (T1) and the number of variables that were omitted incorrectly (T2)
# during the stepwise regression.

T1_vec <- c()

T2_vec <- c()

# Creating a list to record the stepwise regression models.

model_list <- vector(mode = "list", length = length(B_mag_vec))

for (i in 1:length(B_mag_vec)){
  
  # Storing the beta coefficients in a vector.
  
  beta_coeff <- c(rep(B_mag_vec[i],num_lin_var_vec[i])
                  ,rep(0,num_var_vec[i]-num_lin_var_vec[i])
                  )
  
  # Creating the explanatory variables.
  
  x <- matrix(rnorm(n_obs*num_var_vec[i]), nrow = n_obs) 
  
  # Creating the response variable.
  
  y <- x %*% beta_coeff + matrix(rnorm(n_obs),nrow = n_obs) 
  
  # Combining the response and explanatory variables in a single dataframe.
  
  df_comb <- as.data.frame(cbind(y,x))
  
  # Assigning the column names for the dataset.
  
  colnames(df_comb) <- c('y',paste('x',1:num_var_vec[i],sep = ''))
  
  # Creating a full linear regression model.
  
  model_lm_full <- lm(y ~.,data = df_comb)
  
  # Performing stepwise regression in the backward direction.
  
  model_step_back <- stepAIC(model_lm_full, direction = "backward")
  
  summary(model_lm_full)
  
  summary(model_step_back)
  
  # For estimating the model performance of stepwise regression, it is better to
  # record the number of variables that were retained incorrectly (T1) and the
  # number of variables that were omitted incorrectly (T2). 
  
  # Storing the model attributes in a variable.
  
  model_attr <- as.numeric(gsub("x", "", attr(terms(model_step_back)
                                              ,"term.labels")
                                )
                           )
  
  # Creating a vector with 0 representing a dropped variable and 1 representing 
  # a retained variable. The length of the vector should equal to the number
  # of the explanatory variables.
  
  flag_vec <- replace(rep(0,num_var_vec[i]), model_attr, 1)
  
  # Creating a vector with 0 indicating a variable not having a linear 
  # relationship and 1 having a linear relationship.
  
  flag_lm_vec <- beta_coeff
  
  flag_lm_vec[flag_lm_vec > 0] <- 1
  
  # Storing the number of explanatory variables.
  
  num_var <- num_var_vec[i]
  
  # Storing the number of explanatory variables that are linearly related to Y.
  
  num_lin_var <- num_lin_var_vec[i]
  
  # Initializing T1 and T2.
  
  T1 <- 0
  
  T2 <- 0
  
  if (num_lin_var_vec[i] != 0) {
    
    T2 <- sum(flag_vec[1:num_lin_var] != flag_lm_vec[1:num_lin_var])
    
  }
  
  T1 <- sum (flag_vec[(num_lin_var+1):num_var] != flag_lm_vec[(num_lin_var+1):num_var])
  
  # Storing the T1 and T2 values in their respective vectors.
  
  T1_vec <- c(T1_vec,T1)
  
  T2_vec <- c(T2_vec,T2)
  
  # Storing the stepwise regression model in the list variable.
  
  model_list[[i]] <- model_step_back
  
}

# Storing the simulation results in a dataframe.

result_df <- data.frame(T1 = T1_vec,T2 = T2_vec,beta_mag = B_mag_vec
                        ,num_var = num_var_vec,num_lin_var = num_lin_var_vec
                        )

# Creating a scatterplot to analyze the relationship between number of 
# predictors and T1.

par(mfrow = c(2,2))

plot(result_df$num_var,result_df$T1)

# Calculating the correlation coefficient.

cor(result_df$num_var,result_df$T1)

# A linear model is created using T1 and the number of predictors.
# This line would be fitted onto the scatterplot.

abline(lm(result_df$T1~result_df$num_var), col = 'red')

# Creating a scatterplot to analyze the relationship between number of 
# predictors and T2.

plot(result_df$num_var,result_df$T2)

# Calculating the correlation coefficient.

cor(result_df$num_var,result_df$T2)

# A linear model is created using T2 and the number of predictors.
# This line would be fitted onto the scatterplot.

abline(lm(result_df$T2~result_df$num_var), col = 'blue')

# The average T1 and T2 is calculated for each number of predictors.

result_df_grp <- result_df %>% group_by(num_var) %>% summarize(T1 = mean(T1)
                                                               ,T2 = mean(T2)
                                                               )

result_df_grp <- as.data.frame(result_df_grp)

# Creating a scatterplot to analyze the relationship between number of 
# predictors and T1.

plot(result_df_grp$num_var,result_df_grp$T1)

# Calculating the correlation coefficient.

cor(result_df_grp$num_var,result_df_grp$T1)

# A linear model is created using T1 and the number of predictors.
# This line would be fitted onto the scatterplot.

abline(lm(result_df_grp$T1~result_df_grp$num_var), col = 'red')

# Creating a scatterplot to analyse the relationship between number of 
# predictors and T2.

plot(result_df_grp$num_var,result_df_grp$T2)

# Calculating the correlation coefficient.

cor(result_df_grp$num_var,result_df_grp$T2)

# A linear model is created using T2 and the number of predictors.
# This line would be fitted onto the scatterplot.

abline(lm(result_df_grp$T2~result_df_grp$num_var), col = 'blue')

par(mfrow = c(1,1))

# From the plots we can see that as the number of predictors increases, the 
# performance of step-wise regression also increases.
