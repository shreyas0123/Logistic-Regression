#################################### problem1 #########################################################
library(readr)

#load the dataset
Affairs_data <- read.csv("C:\\Users\\DELL\\Downloads\\Affairs.csv")
summary(Affairs_data)
str(Affairs_data)
summary(Affairs_data$naffairs)

#Performing EDA
#convering continous data into discrete >=6 make 1 else 0 for naffairs
Affairs_data$naffairs <- ifelse(Affairs_data$naffairs >= 6 , 1,0)

summary(Affairs_data)
str(Affairs_data)

#Drop the column X
Affairs_data <- Affairs_data[-1]

#Checking NA values
sum(is.na(Affairs_data))

colnames(Affairs_data)

# Preparing a linear regression 
mod_lm <- lm(naffairs ~ ., data = Affairs_data)
summary(mod_lm)

pred1 <- predict(mod_lm, Affairs_data)
pred1

# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
model <- glm(naffairs ~ ., data = Affairs_data, family = "binomial")
summary(model)
# We are going to use NULL and Residual Deviance to compare the between different models

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Prediction to check model validation
prob <- predict(model, Affairs_data, type = "response")
prob

# or use plogis for prediction of probabilities
prob <- plogis(predict(model, Affairs_data))

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0.5, Affairs_data$naffairs)
confusion

# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion))
Acc

# Convert the probabilities to binary output form using cutoff
pred_values <- ifelse(prob > 0.5, 1, 0)

library(caret)
# Confusion Matrix
confusionMatrix(factor(Affairs_data$naffairs, levels = c(0, 1)), factor(pred_values, levels = c(0, 1)))

# Build Model on 100% of data
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
fullmodel <- glm(naffairs ~ ., data = Affairs_data, family = "binomial")
summary(fullmodel)

prob_full <- predict(fullmodel, Affairs_data, type = "response")
prob_full

# Decide on optimal prediction probability cutoff for the model
install.packages("InformationValue")
library(InformationValue)
optCutOff <- optimalCutoff(Affairs_data$naffairs, prob_full)
optCutOff

# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(Affairs_data$naffairs, prob_full, threshold = optCutOff)

# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(Affairs_data$naffairs, prob_full)

# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)

results <- confusionMatrix(predvalues, Affairs_data$naffairs)

sensitivity(predvalues, Affairs_data$naffairs)
confusionMatrix(actuals = Affairs_data$naffairs, predictedScores = predvalues)

###################
# Data Partitioning
n <- nrow(Affairs_data)
n1 <- n * 0.85
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- Affairs_data[train_index, ]
test <- Affairs_data[-train_index, ]

# Train the model using Training data
finalmodel <- glm(naffairs ~ ., data = train, family = "binomial")
summary(finalmodel)

# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test

# Confusion matrix 
confusion <- table(prob_test > optCutOff, test$naffairs)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optCutOff, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values

table(test$naffairs, test$pred_values)

# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train

# Confusion matrix
confusion_train <- table(prob_train > optCutOff, train$naffairs)
confusion_train

# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train

################################ problem2 #####################################################
library(readr)
# Load the Dataset
advertising_data <- read.csv("C:\\Users\\DELL\\Downloads\\advertising.csv")

#EDA
sum(is.na(advertising_data))
colnames(advertising_data)

#label encoding
factors <- factor(advertising_data$Ad_Topic_Line)
advertising_data$Ad_Topic_Line <- as.numeric(factors)

factors <- factor(advertising_data$City)
advertising_data$City <- as.numeric(factors)

factors <- factor(advertising_data$Country)
advertising_data$Country <- as.numeric(factors)

factors <- factor(advertising_data$Timestamp)
advertising_data$Timestamp <- as.numeric(factors)

str(advertising_data)

# Preparing a linear regression 
mod_lm <- lm(Clicked_on_Ad ~ ., data = advertising_data)
summary(mod_lm)

pred1 <- predict(mod_lm, advertising_data)
pred1

# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
model <- glm(Clicked_on_Ad ~ ., data = advertising_data, family = "binomial")
summary(model)

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Prediction to check model validation
prob <- predict(model, advertising_data, type = "response")
prob

# or use plogis for prediction of probabilities
prob <- plogis(predict(model, advertising_data))

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0.5, advertising_data$Clicked_on_Ad)
confusion

# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion))
Acc

# Convert the probabilities to binary output form using cutoff
pred_values <- ifelse(prob > 0.5, 1, 0)

library(caret)
# Confusion Matrix
confusionMatrix(factor(advertising_data$Clicked_on_Ad, levels = c(0, 1)), factor(pred_values, levels = c(0, 1)))

# Build Model on 100% of data
advertising_data <- advertising_data
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
fullmodel <- glm(Clicked_on_Ad ~ ., data = advertising_data, family = "binomial")
summary(fullmodel)

prob_full <- predict(fullmodel, advertising_data, type = "response")
prob_full

# Decide on optimal prediction probability cutoff for the model
library(InformationValue)
optCutOff <- optimalCutoff(advertising_data$Clicked_on_Ad, prob_full)
optCutOff

# Check multicollinearity in the model
library(car)
vif(fullmodel)

# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(advertising_data$Clicked_on_Ad, prob_full, threshold = optCutOff)


# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(advertising_data$Clicked_on_Ad, prob_full)


# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)

results <- confusionMatrix(predvalues, advertising_data$Clicked_on_Ad)

sensitivity(predvalues, advertising_data$Clicked_on_Ad)
confusionMatrix(actuals = advertising_data$Clicked_on_Ad, predictedScores = predvalues)


###################
# Data Partitioning
n <- nrow(advertising_data)
n1 <- n * 0.85
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- advertising_data[train_index, ]
test <- advertising_data[-train_index, ]

# Train the model using Training data
finalmodel <- glm(Clicked_on_Ad ~ ., data = train, family = "binomial")
summary(finalmodel)

# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test

# Confusion matrix 
confusion <- table(prob_test > optCutOff, test$Clicked_on_Ad)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optCutOff, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values

table(test$Clicked_on_Ad, test$pred_values)


# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train

# Confusion matrix
confusion_train <- table(prob_train > optCutOff, train$Clicked_on_Ad)
confusion_train

# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train

#################################### problem3 #################################################
# Load the Dataset
Candidate_data <- read.csv("C:\\Users\\DELL\\Downloads\\election_data.csv") 
sum(is.na(Candidate_data))
summary(Candidate_data$Result)

#EDA
# Omitting NA values from the Data 
Candidate_data <- na.omit(Candidate_data) # na.omit => will omit the rows which has atleast 1 NA value
dim(Candidate_data)

sum(is.na(Candidate_data))
dim(Candidate_data)
###########

colnames(Candidate_data)

# Preparing a linear regression 
mod_lm <- lm(Result ~ ., data = Candidate_data)
summary(mod_lm)

pred1 <- predict(mod_lm, Candidate_data)
pred1

# We can also include NA values but where ever it finds NA value
# probability values obtained using the glm will also be NA 
# So they can be either filled using imputation technique or
# exlclude those values 


# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
model <- glm(Result ~ ., data = Candidate_data, family = "binomial")
summary(model)
# We are going to use NULL and Residual Deviance to compare the between different models

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Prediction to check model validation
prob <- predict(model, Candidate_data, type = "response")
prob

# or use plogis for prediction of probabilities
prob <- plogis(predict(model, Candidate_data))

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0.5, Candidate_data$Result)
confusion

# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion))
Acc

# Convert the probabilities to binary output form using cutoff
pred_values <- ifelse(prob > 0.5, 1, 0)

library(caret)
# Confusion Matrix
confusionMatrix(factor(Candidate_data$Result, levels = c(0, 1)), factor(pred_values, levels = c(0, 1)))

# Build Model on 100% of data
Candidate_data <- Candidate_data
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
fullmodel <- glm(Result ~ ., data = Candidate_data, family = "binomial")
summary(fullmodel)

prob_full <- predict(fullmodel, Candidate_data, type = "response")
prob_full

# Decide on optimal prediction probability cutoff for the model
library(InformationValue)
optCutOff <- optimalCutoff(Candidate_data$Result, prob_full)
optCutOff

# Check multicollinearity in the model
library(car)
vif(fullmodel)

# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(Candidate_data$Result, prob_full, threshold = optCutOff)


# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(Candidate_data$Result, prob_full)


# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)

results <- confusionMatrix(predvalues, Candidate_data$Result)

sensitivity(predvalues, Candidate_data$Result)
confusionMatrix(actuals = Candidate_data$Result, predictedScores = predvalues)


###################
# Data Partitioning
n <- nrow(Candidate_data)
n1 <- n * 0.85
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- Candidate_data[train_index, ]
test <- Candidate_data[-train_index, ]

# Train the model using Training data
finalmodel <- glm(Result ~ ., data = train, family = "binomial")
summary(finalmodel)

# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test

# Confusion matrix 
confusion <- table(prob_test > optCutOff, test$Result)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optCutOff, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values

table(test$Result, test$pred_values)


# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train

# Confusion matrix
confusion_train <- table(prob_train > optCutOff, train$Result)
confusion_train

# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train

################################## problem4 ##############################################
# Load the Dataset
Bank_data <- read.csv("C:\\Users\\DELL\\Downloads\\bank_data.csv") 
sum(is.na(Bank_data))

dim(Bank_data)
###########

colnames(Bank_data)

# Preparing a linear regression 
mod_lm <- lm(y ~ ., data = Bank_data)
summary(mod_lm)

pred1 <- predict(mod_lm, Bank_data)
pred1

# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
model <- glm(y ~ ., data = Bank_data, family = "binomial")
summary(model)

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Prediction to check model validation
prob <- predict(model, Bank_data, type = "response")
prob

# or use plogis for prediction of probabilities
prob <- plogis(predict(model, Bank_data))

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0.5, Bank_data$y)
confusion

# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion))
Acc

# Convert the probabilities to binary output form using cutoff
pred_values <- ifelse(prob > 0.5, 1, 0)

library(caret)
# Confusion Matrix
confusionMatrix(factor(Bank_data$y, levels = c(0, 1)), factor(pred_values, levels = c(0, 1)))


# Build Model on 100% of data
Bank_data <- Bank_data
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
fullmodel <- glm(y ~ ., data = Bank_data, family = "binomial")
summary(fullmodel)

prob_full <- predict(fullmodel, Bank_data, type = "response")
prob_full

# Decide on optimal prediction probability cutoff for the model
library(InformationValue)
optCutOff <- optimalCutoff(Bank_data$y, prob_full)
optCutOff

# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(Bank_data$y, prob_full, threshold = optCutOff)


# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(Bank_data$y, prob_full)


# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)

results <- confusionMatrix(predvalues, Bank_data$y)

sensitivity(predvalues, Bank_data$y)
confusionMatrix(actuals = Bank_data$y, predictedScores = predvalues)


###################
# Data Partitioning
n <- nrow(Bank_data)
n1 <- n * 0.85
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- Bank_data[train_index, ]
test <- Bank_data[-train_index, ]

# Train the model using Training data
finalmodel <- glm(y ~ ., data = train, family = "binomial")
summary(finalmodel)

# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test

# Confusion matrix 
confusion <- table(prob_test > optCutOff, test$y)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optCutOff, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values

table(test$y, test$pred_values)


# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train

# Confusion matrix
confusion_train <- table(prob_train > optCutOff, train$y)
confusion_train

# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train

############################################# END ########################################################
