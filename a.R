# Import libraries

install.packages(c('tidyverse', 'caret'))
library(tidyverse)
library(caret)

# Import data

bank <- read.csv("Data/bank.csv", sep=";")

# Examine data

dim(bank)
names(bank)
glimpse(bank)

# Exploratory data analysis 

bank %>% ggplot(aes(job, y, color = marital)) + geom_point()

## histogram of ages
hist(bank$age)
## boxplot of balances
boxplot(bank$age)
## boxplot of ages per marital status
bank %>% ggplot(aes(marital, age)) + geom_boxplot()
## leads grouped by marital status
bank %>% ggplot(aes(marital)) + geom_bar()
## leads grouped by jobs and outcome percentages shown for each job
bank %>% ggplot(aes(job, fill = y)) + geom_bar(position = "stack") + coord_flip() # dodge to shown side by side and fill to show as percentage

## how many calls where made each month
bank %>% group_by(month) %>% data.frame(month = month, calls = n())

bank %>% ggplot(aes(age, balance, color = y)) + geom_boxplot() + facet_wrap(~month)
bank %>% ggplot(aes(age, balance, size = duration, color = y)) + geom_point() + facet_wrap(~job)

set.seed(1, sample.kind = "Rounding")

# Prepare training data

## 70% of data is for training and rest is for testing
train_index <- createDataPartition(bank$y, times = 1, p = 0.7, list = FALSE)
train <- bank[train_index,]
test <- bank[-train_index,]

## prcomp(bank)

# Fitting model

## Use random forest to determine variable importance
rf_fit <- train %>% train(y ~ ., data = ., method = "rf")
## It is seen that duration is a very important predictor
varImp(rf_fit)
## Accuracy is highest when mtry is 22
plot(rf_fit)

# Fitting linear model

glm_fit <- train %>% train(y ~ ., data = ., method = "glm")
glm_y_hat <- predict(glm_fit, test)
mean(glm_y_hat == test$y)

glm_fit_2 <- train %>% train(y ~ balance, data = ., method = "glm")
glm_y_hat_2 <- predict(glm_fit_2, test)
mean(glm_y_hat_2 == test$y)

glm_fit_3 <- train %>% train(y ~ duration + balance, data = ., method = "glm")
glm_y_hat_3 <- predict(glm_fit_3, test)
mean(glm_y_hat_3 == test$y)

glm_fit_4 <- train %>% train(y ~ duration, data = ., method = "glm")
glm_y_hat_4 <- predict(glm_fit_4, test)
mean(glm_y_hat_4 == test$y)

glm_fit_5 <- train %>% train(y ~ poutcome, data = ., method = "glm")
glm_y_hat_5 <- predict(glm_fit_5, test)
mean(glm_y_hat_5 == test$y)

qda_fit <- train %>% train(y ~ duration, data = ., method = "qda")
qda_y_hat <- predict(qda_fit, test)
mean(qda_y_hat == test$y)

