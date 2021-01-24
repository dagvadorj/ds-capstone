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

bank %>% ggplot(aes(age, balance, color = y)) + geom_boxplot() + facet_wrap(~month)
bank %>% ggplot(aes(age, balance, color = y)) + geom_boxplot() + facet_wrap(~month)

set.seed(1, sample.kind = "Rounding")

train_index <- createDataPartition(bank$y, times = 1, p = 0.7, list = FALSE)
train <- bank[train_index,]
test <- bank[-train_index,]

# prcomp(bank)

# bank <- bank %>% mutate(y = factor(y), y_num = ifelse(y == "yes", 1, 0))

# Use random forest to determine variable importance

rf_fit <- train %>% train(y ~ ., data = ., method = "rf")
varImp(rf_fit)
rf_fit
plot(rf_fit)
