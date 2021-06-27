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

str_extract(movielens$title, regex("\\((\\d{4})"))

movielens <- movielens %>% mutate(releaseYear = as.integer(str_extract(movielens$title, regex("(\\d{4}$)"))))

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

## Define predictors



summary(edx)

## Explore correlation between predictors

# edx %>% ggplot(aes(hour, rating)) + geom_tile()

## Divide into train and test sets

set.seed(1, sample.kind = "Rounding")

test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)

train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# TODO Why is 100000 good

train_set_1 <- sample_n(train_set, 100000)
test_set_1 <- sample_n(test_set, 100000)

test_set_1 <- test_set_1 %>%semi_join(train_set_1, by = "movieId") %>% semi_join(train_set_1, by = "userId")

# plot(train_set_1$weekday, train_set_1$rating)

RMSE <- function(predicted, actual) {
  sqrt(mean((predicted - actual)^2))
}

rm(bias_weeks)

memory.limit(9999999999)

y <- train_set_1 %>% 
  select(userId, movieId, rating) %>%
  pivot_wider(names_from = "movieId", values_from = "rating") %>%
  as.matrix()
gc()
y[is.na(y)] <- 0
gc()
pca <- prcomp(y)
gc()

## Effects

## Explore more relevant predictors

## Regularization

## Matrix factorization

## Choose best predictors
# TODO Why these predictors

## Train models

## Test models

mu <- mean(train_set$rating)

bias_movies <- train_set %>%
  group_by(movieId) %>%
  summarize(b_movie = mean(rating - mu))

bias_movies

predicted_ratings <- mu + test_set %>%
  left_join(bias_movies, by = "movieId") %>%
  # mutate(pred = mu + b_movie) %>%
  pull(b_movie)

predicted_ratings

predicted_ratings

test_set$rating

RMSE(predicted_ratings, test_set$rating)

bias_users <- train_set %>% 
  left_join(bias_movies, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_user = mean(rating - mu - b_movie))

bias_users

predicted_ratings <- test_set %>% 
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  mutate(pred = mu + b_movie + b_user) %>%
  pull(pred)
RMSE(predicted_ratings, test_set$rating)

bias_weeks <- train_set %>% 
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  group_by(week) %>%
  summarize(b_week = mean(rating - mu - b_movie - b_user))

bias_weeks[is.na(bias_weeks)] <- 0

predicted_ratings <- test_set %>% 
  left_join(bias_movies, by='movieId') %>%
  left_join(bias_users, by='userId') %>%
  left_join(bias_weeks, by='week') %>%
  mutate(pred = mu + b_movie + b_user + b_week) %>%
  pull(pred)

head(predicted_ratings)

RMSE(predicted_ratings, test_set$rating)

## Cross validate models

## Validate models