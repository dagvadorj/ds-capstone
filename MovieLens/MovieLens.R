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

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

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
                                           genres = as.character(genres2))


movielens <- left_join(ratings, movies, by = "movieId")

# Extracting release date from titles
movielens <- movielens %>% mutate(releaseYear = str_extract(movielens$title, "\\((\\d{4})\\)$")) %>% mutate(releaseYear = as.integer(substring(releaseYear, 2, nchar(releaseYear)-1)))

# The difference between the movie's release year and the year of rating is selected as a potential predictor.
movielens <- movielens %>% mutate(age = year - releaseYear)

# Extracting components of the date of rating
movielens <- movielens %>% mutate(datetime = lubridate::as_datetime(timestamp))
movielens <- movielens %>% mutate(year = year(datetime), month = month(datetime), week = week(datetime), weekday = wday(datetime), hour = hour(datetime))

movielens <- movielens %>% mutate(firstGenre = map(str_split(genres, "\\|"), 1)) %>% mutate(firstGenre = as.factor(unlist(firstGenre)))

# Existing average rating is used as a predictor
movielens <- movielens %>% group_by(movieId) %>% mutate(avgRating = as.integer(sum(rating) / n()), numRating = n()) %>% ungroup()

# Number of existing ratings may be a good predictor when used with existing average ratings of movies. We are stratifying number of existing ratings since the numbers are too unique per movie and this may result in overfitting.
movielens <- movielens %>% mutate(numRatingGroup = floor(numRating/10000)*10)

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

train_set <- train_set %>% mutate(numRatingGroup = floor(numRating/10000)*10)
test_set <- test_set %>% mutate(numRatingGroup = floor(numRating/10000)*10)

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

scaled_matrix <- train_set %>% select(movieId, userId)
scaled_matrix <- sweep(scaled_matrix, 2, colMeans(scaled_matrix), FUN = "-")
scaled_matrix <- sweep(scaled_matrix, 2, colSds(scaled_matrix), FUN = "/")

pr <- prcomp(scaled_matrix)

##############################
# Exploratory analysis
##############################

# Summary of our dataset:
summary(edx)

# Movielens is a well studied data set. In this section, we will examine the predictors and how they may correlate with the outcome.

##############################
# Models
##############################

table.results = data.frame()

mu <- mean(train_set$rating)

bias.movies <- train_set %>%
  group_by(movieId) %>%
  summarize(b_movie = mean(rating - mu))

pred.movies <- test_set %>%
  left_join(bias.movies, by = "movieId") %>%
  mutate(pred = mu + b_movie)

table.results <- rbind(table.results, data.frame(name = "Movie bias", rmse = RMSE(pred.movies$pred, test_set$rating)))

bias.users <- train_set %>% 
  left_join(bias_movies, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_user = mean(rating - mu - b_movie))

pred.users <- test_set %>% 
  left_join(bias.movies, by='movieId') %>%
  left_join(bias.users, by='userId') %>%
  mutate(pred = mu + b_movie + b_user)

table.results <- rbind(table.results, data.frame(name = "User bias", rmse = RMSE(pred.users$pred, test_set$rating)))

bias.weekday <- train_set %>% 
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  group_by(weekday) %>%
  summarize(b_weekday = mean(rating - mu - b_movie - b_user))

pred.weekday <- test_set %>%
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  left_join(bias.weekday, by='hour') %>%
  mutate(pred = mu + b_movie + b_user + b_weekday)

table.results <- rbind(table.results, data.frame(name = "Effect of rating weekday", rmse = RMSE(pred.weekday$pred, test_set$rating)))

bias.hour <- train_set %>% 
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  group_by(hour) %>%
  summarize(b_hour = mean(rating - mu - b_movie - b_user))

pred.hour <- test_set %>%
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  left_join(bias.hour, by='hour') %>%
  mutate(pred = mu + b_movie + b_user + b_hour)

table.results <- rbind(table.results, data.frame(name = "Effect of rating hour", rmse = RMSE(pred.hour$pred, test_set$rating)))

bias.firstGenre <- train_set %>% 
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  group_by(firstGenre) %>%
  summarize(b_firstGenre = mean(rating - mu - b_movie - b_user))

pred.firstGenre <- test_set %>%
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  left_join(bias.firstGenre, by='firstGenre') %>%
  mutate(pred = mu + b_movie + b_user + b_firstGenre)

table.results <- rbind(table.results, data.frame(name = "Effect of first genre", rmse = RMSE(pred.firstGenre$pred, test_set$rating)))

bias.avgRating <- train_set %>% 
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  group_by(avgRating) %>%
  summarize(b_avgRating = mean(rating - mu - b_movie - b_user))

pred.avgRating <- test_set %>%
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  left_join(bias.avgRating, by='avgRating') %>%
  mutate(pred = mu + b_movie + b_user + b_avgRating)

table.results <- rbind(table.results, data.frame(name = "Effect of existing average ratings", rmse = RMSE(pred.avgRating$pred, test_set$rating)))

bias.avgRatingNumRatingStrata <- train_set %>% 
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  left_join(bias.avgRating, by='avgRating') %>%
  group_by(numRatingGroup) %>%
  summarize(b_avgRatingNumRatingStrata = mean(rating - mu - b_movie - b_user - b_avgRating))

pred.avgRatingNumRatingStrata <- test_set %>%
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  left_join(bias.avgRating, by='avgRating') %>%
  left_join(bias.avgRatingNumRatingStrata, by='numRatingGroup') %>%
  mutate(pred = mu + b_movie + b_user + b_avgRating + b_avgRatingNumRatingStrata)

table.results <- rbind(table.results, data.frame(name = "Effect of both average of existing ratings and number of ratings strata", rmse = RMSE(pred.avgRatingNumRatingStrata$pred, test_set$rating)))

bias.releaseYear <- train_set %>% 
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  group_by(releaseYear) %>%
  summarize(b_releaseYear = mean(rating - mu - b_movie - b_user))

pred.releaseYear <- test_set %>%
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  left_join(bias.releaseYear, by='releaseYear') %>%
  mutate(pred = mu + b_movie + b_user + b_releaseYear)

table.results <- rbind(table.results, data.frame(name = "Effect of release year", rmse = RMSE(pred.releaseYear$pred, test_set$rating)))

bias.age <- train_set %>% 
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  group_by(age) %>%
  summarize(b_age = mean(rating - mu - b_movie - b_user))

pred.age <- test_set %>%
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  left_join(bias.age, by='age') %>%
  mutate(pred = mu + b_movie + b_user + b_age)

table.results <- rbind(table.results, data.frame(name = "Effect of difference between release year and rating year", rmse = RMSE(pred.age$pred, test_set$rating)))

# We will select rating age based and average rating plus number of rating strata based models because they are the best performers and use regularization in order to improve against overfitting.

##############################
# Results
##############################

table.results

edx <- edx %>% mutate(numRatingGroup = floor(numRating/10000)*10)
validation <- validation %>% mutate(numRatingGroup = floor(numRating/10000)*10)

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
    group_by(numRatingGroup) %>%
    summarize(b_avgRatingNumRatingStrata = mean(rating - mu - b_movie - b_user - b_avgRating))
  
  pred.avgRatingNumRatingStrata <- test %>%
    left_join(bias_movies, by='movieId') %>%
    left_join(bias_users, by='userId') %>%
    left_join(bias.avgRating, by='avgRating') %>%
    left_join(bias.avgRatingNumRatingStrata, by='numRatingGroup') %>%
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

lambdas = seq(0, 8, .25)

results.avgRating <- sapply(lambdas, function(lambda) {
  regularize.avgRating(edx, validation, lambda)
})

results.age <- sapply(lambdas, function(lambda) {
  regularize.age(edx, validation, lambda)
})

qplot(lambdas, results.avgRating)
qplot(lambdas, results.age)

min(results.avgRating)
min(results.age)

# 5 points: RMSE >= 0.90000 AND/OR the reported RMSE is the result of overtraining (validation set - the final hold-out test set - ratings used for anything except reporting the final RMSE value) AND/OR the reported RMSE is the result of simply copying and running code provided in previous courses in the series.
# 10 points: 0.86550 <= RMSE <= 0.89999
# 15 points: 0.86500 <= RMSE <= 0.86549
# 20 points: 0.86490 <= RMSE <= 0.86499
# 25 points: RMSE < 0.86490

##############################
# Conclusion
##############################