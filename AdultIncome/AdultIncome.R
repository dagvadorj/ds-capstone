## 1 Executive summary

# Thank you Mr. Irizarry and his team as well as the peers for the great opportunity of learning and validating my understanding of machine learning.

## 2 Data preparation

### 2.1 Loading data

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(GGally)
library(ggridges)
library(ggthemes)
library(rpart.plot)
library(MASS)

incomes <- fread(text = gsub("::", "\t", readLines(unzip("archive.zip", "adult.csv"))), col.names = c("age", "workclass", "fnlwgt", "education", "education.number", "marital.status", "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss", "hours.per.week", "native.country", "income"))

str(incomes)
dim(incomes)

# When we glance the data we see that some entries have values noted as "?" where the value is not available. We will get rid of these data.

colSums(incomes == "?")

# We see that workclass, occupation, and native.country columns have "?" values.

incomes <- incomes %>% filter(!(workclass == "?" | occupation == "?" | native.country == "?"))

dim(incomes)

colSums(incomes == "?")

# As for the predictors I decide not to use fnlwgt which is a weight that accounts for socio-economic and demographic features of individuals as calculated by CPS [2]. We are already trying to understand the effects of socio-economic and demographic features in the data set (such as age, work class, education, etc.) for the income. 

# Furthermore, the capital gain and capital loss predictors do not seem to characterize the data very well as there is no balance or diversification accross our data. Hence, we will not be using these predictors as well.

hist(incomes$capital.loss)
hist(incomes$capital.loss)

# Quick glance at education and education.number predictors makes it easy to see that the education.number is a one-to-one numerical representation of education. I will use education for exploratory data analysis and education.number for model fitting since education is more user-friendly because it is easilty readable and education.number is numeric and also shows the degree of education.

incomes %>% arrange(education.number) %>% ggplot(aes(education, education.number)) + geom_point()

incomes %>% ggplot(aes(marital.status, income, group = workclass)) + geom_boxplot()

incomes <- incomes %>% select(-capital.gain, -capital.loss, -fnlwgt)

### 2.2 Preparing data types

# When we examine the data, we can see that some character data need to be converted to factor data type.

summary(incomes)

# For example, the workclass column consists of selection among few choices.

incomes %>% ggplot(aes(workclass)) + geom_histogram(stat = "count")
unique(incomes$workclass)

# We use the mutate function to convert the data types where necessary.

incomes <- incomes %>% mutate(workclass = as.factor(workclass), education = as.factor(education), marital.status = as.factor(marital.status), occupation = as.factor(occupation), relationship = as.factor(relationship), race = as.factor(race), sex = as.factor(sex), native.country = as.factor(native.country), income = as.factor(income))

# Let's examine our data set one more time.

summary(incomes)

# We see that there is no N/A or 0 data to clean in our data set using the colSums function.

colSums(is.na(incomes))
colSums(incomes == 0)

### 2.3 Preparing the test and train sets

set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(incomes$income, times = 1, p = 0.2, list = FALSE)
train_set <- incomes[-test_index,]
test_set <- incomes[test_index,]

# Furthermore, let us set the fraction point to a fixed number so that the model accuracies can be easily compared.

options(digits = 5)

## 3 Exploratory data analysis

for (j in 1:ncol(train_set)) {
  colname <- as.character(colnames(train_set)[j])
  print(train_set %>% ggplot(aes_string(colname)) + geom_histogram(stat = "count", color = "pink"))
}

incomes %>% ggplot(aes(occupation, fill = income)) + geom_bar()

## 4 Methods

# We can examine significant statistical figures of the column in incomes data set using the summary function.

summary(incomes)

### 4.1 Linear regression

# Now we have 9 potential predictors. Our goal is to select the most meaningful predictors for building the best model. One way of doing this is to use step wise algorithm to test out the predictors. There are two kinds of stepwise search algorithm - backward search and forward search.

# The backward search algorithm starts from a model that accounts for all predictors and tries to remove predictors one by one while not decreasing the quality of the model represented by AIC [3].

model.full <- glm(income ~ age + workclass + education.number + marital.status + occupation + relationship + race + sex + hours.per.week + native.country, data = train_set, family="binomial")

model.step.backward <- stepAIC(model.full, direction = "backward")
model.step.backward

# The backward search algorithm removes only the race predictor and leaves out the other eight predictors: age + workclass + education.number + marital.status + occupation + relationship + sex + hours.per.week + native.country

# The forward search algorithm starts from a model without any predictors and tries to add predictors one by one while increasing the quality of the model represented by AIC [3].

model.step.forward <- stepAIC(glm(income ~ 1, data = train_set, family="binomial"), direction = "forward", scope = income ~ age + workclass + education.number + marital.status + occupation + relationship + race + sex + hours.per.week + native.country)
model.step.forward

# The forward search algorithm omits only the race predictor and adds other eight predictors: age + workclass + education.number + marital.status + occupation + relationship + sex + hours.per.week + native.country just like the backward search algorithm.

# We will check out the accuracy of linear regression model with abovementioned predictors.

model.lm0 <- train_set %>% train(income ~ age + workclass + education.number + marital.status + occupation + relationship + sex + hours.per.week + native.country, data = ., method = "glm")
model.lm0
pred.lm0 <- predict(model.lm0, test_set)
mean(pred.lm0 == test_set$income)

# 0.82778

# Unfortunately, eight predictors are too many and we will try to fit other models.

### 4.2 Other models

# Decision trees are a good way to understand how and in what order the output is affected by the predictors. There are several ways to construct
# decision trees that account for different aspects of the predictors, their relations, and independent natures.
# Recursive partitioning can be used to understand the importance of the predictors. The great thing about the recursive partitioning is that it recursively try out different orders of the predictors in order to come up with the best accuracy. 
# The caret package includes train function can is capable of training data set using different algorithms with different tuning options. Here I will use first the rpart algorithm to construct and study the predictors and try to understand which predictor(s) have more effect on the output.

model.rpart <- train_set %>% train(income ~ age + workclass + education.number + marital.status + occupation + relationship + race + sex + hours.per.week + native.country, data = ., method = "rpart")

pred.rpart <- predict(model.rpart, test_set)
mean(pred.rpart == test_set$income)

# 0.82032

# According to rpart model, the importance of the predictors for the output are:

varImp(model.rpart, scale = FALSE)

# From here we understand that the education, marital status, age,  and hours per week have a higher level of importance.

rpart.plot(model.rpart$finalModel)
model.rpart$results

# We note that tuning the linear regression model by modifying the predictors will not give us a better accuracy than .82778.

model.lm <- train_set %>% train(income ~ education.number + marital.status + age + hours.per.week, data = ., method = "glm")
model.lm
pred.lm <- predict(model.lm, test_set)
mean(pred.lm == test_set$income)

# 0.81551

model.lm <- train_set %>% train(income ~ education.number + marital.status + age, data = ., method = "glm")
model.lm
coef(model.lm)
pred.lm <- predict(model.lm, test_set)
mean(pred.lm == test_set$income)

# 0.81717

model.lm <- train_set %>% train(income ~ education.number + marital.status, data = ., method = "glm")
pred.lm <- predict(model.lm, test_set)
mean(pred.lm == test_set$income)

# 0.81717

model.lm <- train_set %>% train(income ~ education.number + occupation + hours.per.week * education.number, data = ., method = "glm")
pred.lm <- predict(model.lm, test_set)
mean(pred.lm == test_set$income)

# Now let's start examining other models by using the variables of importance. K-nearest neighbors algorithm is good for examining multi-dimensional data set like ours.

model.knn0 <- train_set %>% train(income ~ education.number + marital.status + age + hours.per.week, data = ., method = "knn")
model.knn0
pred.knn0 <- predict(model.knn0, test_set)
mean(pred.knn0 == test_set$income)

# 0.80491

# We also note that using other sets of the important variables produce slightly better accuracy.

model.knn1 <- train_set %>% train(income ~ education.number + marital.status + age, data = ., method = "knn")
model.knn1
pred.knn1 <- predict(model.knn1, test_set)
mean(pred.knn1 == test_set$income)

# 0.81551

model.knn2 <- train_set %>% train(income ~ education.number + marital.status + occupation, data = ., method = "knn")
model.knn2
pred.knn2 <- predict(model.knn2, test_set)
mean(pred.knn2 == test_set$income)

# 0.82065

# We will use two more models to try to come up with a better accuracy.

model.lda <- train_set %>% train(income ~ education.number + marital.status + age, data = ., method = "lda")
model.lda
pred.lda <- predict(model.lda, test_set)
mean(pred.lda == test_set$income)

# 0.81518

model.qda <- train_set %>% train(income ~ education.number + marital.status + age, data = ., method = "qda")
model.qda
pred.qda <- predict(model.qda, test_set)
mean(pred.qda == test_set$income)
confusionMatrix(pred.qda, test_set$income) # TODO mind for specificity and sensitivity

# 0.73181

## Results

as.numeric(confusionMatrix(pred.lm, test_set$income)$byClass["Sensitivity"])

table.results <- data.frame()
table.results <- rbind(table.results, data.frame(name = "Linear regression", accuracy = mean(pred.lm0 == test_set$income), sensitivity = as.numeric(confusionMatrix(pred.lm0, test_set$income)$byClass["Sensitivity"]), specificity = as.numeric(confusionMatrix(pred.lm0, test_set$income)$byClass["Specificity"])))
table.results <- rbind(table.results, data.frame(name = "Recursive partitioning", accuracy = mean(pred.rpart == test_set$income), sensitivity = as.numeric(confusionMatrix(pred.rpart, test_set$income)$byClass["Sensitivity"]), specificity = as.numeric(confusionMatrix(pred.rpart, test_set$income)$byClass["Specificity"])))
table.results <- rbind(table.results, data.frame(name = "KNN 0", accuracy = mean(pred.knn0 == test_set$income), sensitivity = as.numeric(confusionMatrix(pred.knn0, test_set$income)$byClass["Sensitivity"]), specificity = as.numeric(confusionMatrix(pred.knn0, test_set$income)$byClass["Specificity"])))
table.results <- rbind(table.results, data.frame(name = "KNN 1", accuracy = mean(pred.knn1 == test_set$income), sensitivity = as.numeric(confusionMatrix(pred.knn1, test_set$income)$byClass["Sensitivity"]), specificity = as.numeric(confusionMatrix(pred.knn1, test_set$income)$byClass["Specificity"])))
table.results <- rbind(table.results, data.frame(name = "KNN 2", accuracy = mean(pred.knn2 == test_set$income), sensitivity = as.numeric(confusionMatrix(pred.knn2, test_set$income)$byClass["Sensitivity"]), specificity = as.numeric(confusionMatrix(pred.knn2, test_set$income)$byClass["Specificity"])))
table.results <- rbind(table.results, data.frame(name = "LDA", accuracy = mean(pred.lda == test_set$income), sensitivity = as.numeric(confusionMatrix(pred.lda, test_set$income)$byClass["Sensitivity"]), specificity = as.numeric(confusionMatrix(pred.lda, test_set$income)$byClass["Specificity"])))
table.results <- rbind(table.results, data.frame(name = "QDA", accuracy = mean(pred.qda == test_set$income), sensitivity = as.numeric(confusionMatrix(pred.qda, test_set$income)$byClass["Sensitivity"]), specificity = as.numeric(confusionMatrix(pred.qda, test_set$income)$byClass["Specificity"])))

table.results

## Conclusion

# We note that however it has has too many predictors, the linear regression model with eight predictors performed the best. We have tried to beat this model using KNN, QDA, and LDA models - models that do well with many predictors. However, in the end the linear regression model has the best accuracy. Moreover, the linear model has similar sensitivities and specificities with the other models. In other words, the model is not lagging from the other models in this area as well. If our data set included many numeric predictors in other words if the important socio-economic and demographic predictors were numeric, we would have chance to implement different analyses of clustering, matrix factorization, and component analysis as we learned in the course. I am looking forward to implement these methods and analysis for other data sets in the future. I would like to again thank Mr.Irizarry and his staff for the great opportunity.

## References

# [1] Irizarry. Rafael. Introduction to Data Science. 2019, found at https://leanpub.com/datasciencebook
# [2] https://www.kaggle.com/uciml/adult-census-income
# [3] Dalpiaz. David, Applied Statistics with R. 2021
