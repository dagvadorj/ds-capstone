##############################
# Executive summary
##############################

# Movie Lens database is examined and models were reviewed and fit for the Hardvard edX Data Science: Capstone project.

# The initial section of the code is based on the boilerplate code provided at the "Create Train and Final Hold-out Test Sets" section of the course found at https://learning.edx.org/course/course-v1:HarvardX+PH125.9x+1T2021/block-v1:HarvardX+PH125.9x+1T2021+type@sequential+block@e8800e37aa444297a3a2f35bf84ce452/block-v1:HarvardX+PH125.9x+1T2021+type@vertical+block@e9abcdd945b1416098a15fc95807b5db. Following columns were added to the movielens data frame in order to be used as potential predictors: releaseYear, ratingAge, year, month, week, weekday, hour, avgRating, firstGenre.

# Different models were used by utilizing train method from the caret library. However, every try took unfeasible amount of time in my computer with 16 gig memory. Therefore linear model provided at the "Regularization" section of the Data Science: Machine Learning found at https://learning.edx.org/course/course-v1:HarvardX+PH125.8x+2T2020/block-v1:HarvardX+PH125.8x+2T2020+type@sequential+block@a5bcc5177a5b440eb3e774335916e95d/block-v1:HarvardX+PH125.8x+2T2020+type@vertical+block@0f5cd79d0f374106a640b63f2c82d56a that examines accumulative biases of predictors were used.

##############################
# Data preparation
##############################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")
install.packages(c("ggridges"))

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(GGally)
library(ggridges)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))), col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

# In order to save time, this needs to be run once the dataset zip is donwloaded already
ratings <- fread(text = gsub("::", "\t", readLines("ml-10M100K/ratings.dat")), col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines("ml-10M100K/movies.dat"), "\\::", 3)

colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Extracting release date from titles
movielens <- movielens %>% mutate(releaseYear = str_extract(movielens$title, "\\((\\d{4})\\)$")) %>% mutate(releaseYear = as.integer(substring(releaseYear, 2, nchar(releaseYear)-1)))

# Extracting components of the date of rating
movielens <- movielens %>% mutate(datetime = lubridate::as_datetime(timestamp))
movielens <- movielens %>% mutate(year = year(datetime), month = month(datetime), week = week(datetime), weekday = wday(datetime), hour = hour(datetime))

# The difference between the movie's release year and the year of rating is selected as a potential predictor.
movielens <- movielens %>% mutate(age = year - releaseYear)

# Existing average rating is used as a predictor
movielens <- movielens %>% group_by(movieId) %>% mutate(avgRating = as.integer(sum(rating) / n()), numRating = n()) %>% ungroup()

# Number of existing ratings may be a good predictor when used with existing average ratings of movies. We are stratifying number of existing ratings since the numbers are too unique per movie and this may result in overfitting.
movielens <- movielens %>% mutate(numRatingStrata = floor(numRating/10000)*10)

movielens$numRatingStrata

# We will not be using datetime, genres, and timestamp as predictors
movielens <- movielens %>% select(-datetime, -genres, -timestamp)

##############################
# Training and testing data set preparation
##############################

summary(movielens)

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Freeing up memory and workspace
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# We will use RMSE function to examine the accuracy of our models since we are dealing with continuous outcomes.
RMSE <- function(predicted, actual) {
  sqrt(mean((predicted - actual)^2, na.rm = TRUE))
}

## Divide into train and test sets

set.seed(1, sample.kind = "Rounding")

test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)

train_set <- edx[-test_index,]
test_set <- edx[test_index,]

test_set <- test_set %>%semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")

##############################
# Selection of methodology
##############################

# Using caret training models are not feasible for this exercise as simple linear regression model for two predictors in train_set data set are taking approximately 19 minutes. Using other models like recursive partitioning and random forest models give insufficient memory errors or run hours when memory usage limit is increased by memory.limit(9999999999).

# model.lm <- train_set %>% train(rating ~ movieId + userId, data = ., method = "glm")

# Training linear regression on a fraction of the train_set also yielded unfeasibly long runtimes for this practice.

# train_set_1 <- sample_n(train_set, 100000)
# test_set_1 <- sample_n(test_set, 100000)

# test_set_1 <- test_set_1 %>%semi_join(train_set_1, by = "movieId") %>% semi_join(train_set_1, by = "userId")
# model.lm <- train_set_1 %>% train(rating ~ movieId + userId, data = ., method = "glm")

##############################
# Exploratory analysis
##############################

# Summary of our dataset using the "summary" method will give us idea about statistical information about each predictor.
summary(edx)

# Movielens is a well studied data set. In this section, we will examine the predictors and how they may correlate with the outcome.

ggcorr(edx, label = TRUE, label_alpha = TRUE)

# As edx is a relatively large dataset, it will consume a lot of time doing exploratory analysis. As such we will be performing the analysis and the model fitting based on train_set.

ggcorr(train_set, label = TRUE, label_alpha = TRUE)

# Indeed the correlation analysis on train_set indicates that

# Of course there are significant correlations between the number of ratings and number of ratings strata, age and release year, month and week, movie and year, since these data are based on one another. Correlations that needs to be noted are average rating and rating, age and rating, number of ratings strata and rating.

# What is the most voted month?

train_set %>% ggplot(aes(week)) + geom_histogram(color = "black") + facet_grid(~rating)

# This analysis indicates that people tend to rate movies more with full rates rather than half rates as well as that people rated more during the holiday season.

train_set %>% ggplot(aes(x = rating, y = as.factor(weekday), color = weekday)) + geom_density_ridges()

##############################
# Models
##############################

# Course benchmark data consists of 5 digits after period. So it will make more sense to output our data this way.
options(digits = 5)

fit.models <- function(train, test) {

  table.results = data.frame()
  
  mu <- mean(train$rating)
  
  bias.movies <- train %>%
    group_by(movieId) %>%
    summarize(b_movie = mean(rating - mu))
  
  pred.movies <- test %>%
    left_join(bias.movies, by = "movieId") %>%
    mutate(pred = mu + b_movie)
  
  table.results <- rbind(table.results, data.frame(name = "Movie bias", rmse = RMSE(pred.movies$pred, test$rating)))
  
  bias.users <- train %>% 
    left_join(bias.movies, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_user = mean(rating - mu - b_movie))
  
  pred.users <- test %>% 
    left_join(bias.movies, by='movieId') %>%
    left_join(bias.users, by='userId') %>%
    mutate(pred = mu + b_movie + b_user)
  
  table.results <- rbind(table.results, data.frame(name = "User bias", rmse = RMSE(pred.users$pred, test$rating)))
  
  bias.week <- train %>% 
    left_join(bias.movies, by='movieId') %>%
    left_join(bias.users, by='userId') %>%
    group_by(week) %>%
    summarize(b_week = mean(rating - mu - b_movie - b_user))
  
  pred.week <- test %>%
    left_join(bias.movies, by='movieId') %>%
    left_join(bias.users, by='userId') %>%
    left_join(bias.week, by='week') %>%
    mutate(pred = mu + b_movie + b_user + b_week)
  
  table.results <- rbind(table.results, data.frame(name = "Movie and user biases + effect of week of the rating", rmse = RMSE(pred.week$pred, test$rating)))
  
  bias.hour <- train %>% 
    left_join(bias.movies, by='movieId') %>%
    left_join(bias.users, by='userId') %>%
    group_by(hour) %>%
    summarize(b_hour = mean(rating - mu - b_movie - b_user))
  
  pred.hour <- test %>%
    left_join(bias.movies, by='movieId') %>%
    left_join(bias.users, by='userId') %>%
    left_join(bias.hour, by='hour') %>%
    mutate(pred = mu + b_movie + b_user + b_hour)
  
  table.results <- rbind(table.results, data.frame(name = "Movie and user biases + effect of hour of the rating", rmse = RMSE(pred.hour$pred, test$rating)))
  
  bias.avgRating <- train %>% 
    left_join(bias.movies, by='movieId') %>%
    left_join(bias.users, by='userId') %>%
    group_by(avgRating) %>%
    summarize(b_avgRating = mean(rating - mu - b_movie - b_user))
  
  pred.avgRating <- test %>%
    left_join(bias.movies, by='movieId') %>%
    left_join(bias.users, by='userId') %>%
    left_join(bias.avgRating, by='avgRating') %>%
    mutate(pred = mu + b_movie + b_user + b_avgRating)
  
  table.results <- rbind(table.results, data.frame(name = "Movie and user biases + effect of existing average ratings", rmse = RMSE(pred.avgRating$pred, test$rating)))
  
  bias.avgRatingNumRatingStrata <- train %>% 
    left_join(bias.movies, by='movieId') %>%
    left_join(bias.users, by='userId') %>%
    left_join(bias.avgRating, by='avgRating') %>%
    group_by(numRatingStrata) %>%
    summarize(b_avgRatingNumRatingStrata = mean(rating - mu - b_movie - b_user - b_avgRating))
  
  pred.avgRatingNumRatingStrata <- test %>%
    left_join(bias.movies, by='movieId') %>%
    left_join(bias.users, by='userId') %>%
    left_join(bias.avgRating, by='avgRating') %>%
    left_join(bias.avgRatingNumRatingStrata, by='numRatingStrata') %>%
    mutate(pred = mu + b_movie + b_user + b_avgRating + b_avgRatingNumRatingStrata)
  
  table.results <- rbind(table.results, data.frame(name = "Movie and user biases + effect of both existing average ratings and the strata of number of ratings", rmse = RMSE(pred.avgRatingNumRatingStrata$pred, test$rating)))
  
  bias.releaseYear <- train %>%
    left_join(bias.movies, by='movieId') %>%
    left_join(bias.users, by='userId') %>%
    group_by(releaseYear) %>%
    summarize(b_releaseYear = mean(rating - mu - b_movie - b_user))
  
  pred.releaseYear <- test %>%
    left_join(bias.movies, by='movieId') %>%
    left_join(bias.users, by='userId') %>%
    left_join(bias.releaseYear, by='releaseYear') %>%
    mutate(pred = mu + b_movie + b_user + b_releaseYear)
  
  table.results <- rbind(table.results, data.frame(name = "Movie and user biases + effect of release year", rmse = RMSE(pred.releaseYear$pred, test$rating)))
  
  bias.age <- train %>% 
    left_join(bias.movies, by='movieId') %>%
    left_join(bias.users, by='userId') %>%
    group_by(age) %>%
    summarize(b_age = mean(rating - mu - b_movie - b_user))
  
  pred.age <- test %>%
    left_join(bias.movies, by='movieId') %>%
    left_join(bias.users, by='userId') %>%
    left_join(bias.age, by='age') %>%
    mutate(pred = mu + b_movie + b_user + b_age)
  
  table.results <- rbind(table.results, data.frame(name = "Movie and user biases + effect of difference between release year and rating year", rmse = RMSE(pred.age$pred, test$rating)))

  table.results
}

fit.models(train_set, test_set)

# We will select rating age based and average rating plus number of rating strata based models because they are the best performers and use regularization in order to improve against overfitting.

table.results = fit.models(edx, validation)

##############################
# Results
##############################

regularize.avgRating <- function(train, test, lambda) {

  mu <- mean(train$rating)

  bias_movies <- train %>%
    group_by(movieId) %>%
    summarize(b_movie = sum(rating - mu)/(lambda + n()))

  bias_users <- train %>% 
    left_join(bias_movies, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_user = sum(rating - mu - b_movie)/(lambda + n()))

  bias.avgRating <- train %>% 
    left_join(bias_movies, by='movieId') %>%
    left_join(bias_users, by='userId') %>%
    group_by(avgRating) %>%
    summarize(b_avgRating = mean(rating - mu - b_movie - b_user))

  bias.avgRatingNumRatingStrata <- train %>% 
    left_join(bias_movies, by='movieId') %>%
    left_join(bias_users, by='userId') %>%
    left_join(bias.avgRating, by='avgRating') %>%
    group_by(numRatingStrata) %>%
    summarize(b_avgRatingNumRatingStrata = mean(rating - mu - b_movie - b_user - b_avgRating))
  
  pred.avgRatingNumRatingStrata <- test %>%
    left_join(bias_movies, by='movieId') %>%
    left_join(bias_users, by='userId') %>%
    left_join(bias.avgRating, by='avgRating') %>%
    left_join(bias.avgRatingNumRatingStrata, by='numRatingStrata') %>%
    mutate(pred = mu + b_movie + b_user + b_avgRating + b_avgRatingNumRatingStrata)

  RMSE(pred.avgRatingNumRatingStrata$pred, test$rating)

}

regularize.age <- function(train, test, lambda) {
  
  mu <- mean(train$rating)
  
  bias_movies <- train %>%
    group_by(movieId) %>%
    summarize(b_movie = sum(rating - mu)/(lambda + n()))
  
  bias_users <- train %>% 
    left_join(bias_movies, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_user = sum(rating - mu - b_movie)/(lambda + n()))
  
  bias.age <- train %>% 
    left_join(bias_movies, by='movieId') %>%
    left_join(bias_users, by='userId') %>%
    group_by(age) %>%
    summarize(b_age = mean(rating - mu - b_movie - b_user))

  pred.age <- test %>%
    left_join(bias_movies, by='movieId') %>%
    left_join(bias_users, by='userId') %>%
    left_join(bias.age, by='age') %>%
    mutate(pred = mu + b_movie + b_user + b_age)
  
  RMSE(pred.age$pred, test$rating)
  
}

# Cross-validating lambdas for the selected models
# TODO Why these alphas
lambdas = seq(0, 8, .25)

results.avgRating <- sapply(lambdas, function(lambda) {
  regularize.avgRating(edx, validation, lambda)
})

results.age <- sapply(lambdas, function(lambda) {
  regularize.age(edx, validation, lambda)
})

qplot(lambdas, results.avgRating)
qplot(lambdas, results.age)

# The best performing regularized model yields RMSE of 0.8646

min(results.avgRating)
table.results <- rbind(table.results, data.frame(name = "Effect of existing average ratings + Regularization", rmse = min(results.avgRating)))

# The best performing regularized model yields RMSE of 0.86434

min(results.age)
table.results <- rbind(table.results, data.frame(name = "Effect of difference between release year and rating year + Regularization", rmse = min(results.age)))

##############################
# Conclusion
##############################

table.results

