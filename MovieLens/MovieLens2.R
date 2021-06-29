# Executive summary
# Movie Lens database is examined and models were reviewed and fit for the Hardvard edX Data Science: Capstone project.
# The initial section of the code is based on the boilerplate code provided at the "Create Train and Final Hold-out Test Sets" section of the course found at https://learning.edx.org/course/course-v1:HarvardX+PH125.9x+1T2021/block-v1:HarvardX+PH125.9x+1T2021+type@sequential+block@e8800e37aa444297a3a2f35bf84ce452/block-v1:HarvardX+PH125.9x+1T2021+type@vertical+block@e9abcdd945b1416098a15fc95807b5db. Following columns were added to the movielens data frame in order to be used as potential predictors: releaseYear, ratingAge, year, month, week, weekday, hour, avgRating, firstGenre.
# Different models were used by utilizing train method from the caret library. However, every try took unfeasible amount of time in my computer with 16 gig memory. Therefore linear model that examines accumulative biases of predictors were used.


##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

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
# ratings <- fread(text = gsub("::", "\t", readLines("ml-10M100K/ratings.dat")), col.names = c("userId", "movieId", "rating", "timestamp"))
# movies <- str_split_fixed(readLines("ml-10M100K/movies.dat"), "\\::", 3)

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

movielens <- movielens %>% mutate(releaseYear = str_extract(movielens$title, "\\((\\d{4})\\)$")) %>% mutate(releaseYear = as.integer(substring(releaseYear, 2, nchar(releaseYear)-1)))

movielens <- movielens %>% mutate(datetime = lubridate::as_datetime(timestamp))

movielens <- movielens %>% mutate(year = year(datetime), month = month(datetime), week = week(datetime), weekday = wday(datetime), hour = hour(datetime), firstGenre = map(str_split(genres, "\\|"), 1)) %>% mutate(firstGenre = as.factor(unlist(firstGenre)))

movielens <- movielens %>% group_by(movieId) %>% mutate(avgRating = as.integer(sum(rating) / n())) %>% ungroup()

movielens <- movielens %>% select(-datetime, -genres, -timestamp)

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

## TODO Exploratory analysis

summary(edx)

##

RMSE <- function(predicted, actual) {
  sqrt(mean((predicted - actual)^2, na.rm = TRUE))
}

## Divide into train and test sets

set.seed(1, sample.kind = "Rounding")

test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)

train_set <- edx[-test_index,]
test_set <- edx[test_index,]

test_set <- test_set %>%semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")

# TODO Why is 100000 good

memory.limit(9999999999)

## Effects

## Explore more relevant predictors

## Regularization

## Matrix factorization

## Choose best predictors

## Train models

## Test models

## The logic

mu <- mean(train_set$rating)

bias_movies <- train_set %>%
  group_by(movieId) %>%
  summarize(b_movie = mean(rating - mu))

predicted_ratings <- test_set %>%
  left_join(bias_movies, by = "movieId") %>%
  mutate(pred = mu + b_movie)

RMSE(predicted_ratings$pred, test_set$rating)

bias_users <- train_set %>% 
  left_join(bias_movies, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_user = mean(rating - mu - b_movie))

predicted_ratings <- test_set %>% 
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  mutate(pred = mu + b_movie + b_user)

RMSE(predicted_ratings$pred, test_set$rating)

bias_weeks <- train_set %>% 
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  group_by(avgRating) %>%
  summarize(b_week = mean(rating - mu - b_movie - b_user))

predicted_ratings <- test_set %>%
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  left_join(bias_weeks, by='avgRating') %>%
  mutate(pred = mu + b_movie + b_user + b_week)

RMSE(predicted_ratings$pred, test_set$rating)

## Cross validation regularization

lambdas = seq(0, 10, .1)
lambdas

regularize <- function(train, test, lambda) {

  mu <- mean(train$rating)

  bias_movies <- train %>%
    group_by(movieId) %>%
    summarize(b_movie = sum(rating - mu)/(lambda + n()))

  bias_users <- train %>% 
    left_join(bias_movies, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_user = sum(rating - mu - b_movie)/(lambda + n()))

  bias_weeks <- train %>% 
    left_join(bias_movies, by='movieId') %>%
    left_join(bias_users, by='userId') %>%
    group_by(firstGenre) %>%
    summarize(b_week = sum(rating - mu - b_movie - b_user)/(lambda + n()))
  
  predicted_ratings <- test %>%
    left_join(bias_movies, by='movieId') %>%
    left_join(bias_users, by='userId') %>%
    left_join(bias_weeks, by='firstGenre') %>%
    mutate(pred = mu + b_movie + b_user + b_week)

  RMSE(predicted_ratings$pred, test$rating)

}

dplyr.summarise.inform <- FALSE

results <- sapply(lambdas, function(lambda) {
  regularize(edx, validation, lambda)
})

qplot(lambdas, results)

min(results)

## Validate models