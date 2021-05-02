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
library(RColorBrewer)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
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

# How many threes were given as ratings in the edx dataset?
length(which(edx$rating == 3.0))
edx %>% filter(rating == 3.0) %>% tally()

# How many different movies are in the edx dataset?
length(unique(edx$movieId))
n_distinct(edx$movieId)

n_distinct(edx$userId)

# How many movie ratings are in each of the following genres in the edx dataset?
edx %>% filter(grepl("Romance", genres, fixed = FALSE)) %>% summarize(n = n())

# str_detect
genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

# separate_rows, much slower!
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# Which movie has the greatest number of ratings?
edx %>% group_by(movieId) %>% mutate(num_of_ratings = n(), sum_of_ratings = sum(rating)) %>% ungroup() %>% arrange(desc(num_of_ratings)) %>% select(movieId, title, sum_of_ratings, num_of_ratings) %>% mutate(avg_rating = sum_of_ratings / num_of_ratings) %>% group_by(movieId)

## Define predictors

edx <- edx %>% mutate(datetime = lubridate::as_datetime(timestamp))
edx <- edx %>% mutate(date = date(datetime), year = year(datetime), month = month(datetime), weekday = as.factor(weekdays(datetime)), hour = hour(datetime))

edx <- edx %>% group_by(movieId) %>% mutate(num_of_ratings = n(), sum_of_ratings = sum(rating)) %>% mutate(avg_rating = sum_of_ratings / num_of_ratings) %>% ungroup()

edx <- edx %>% mutate(movieId = as.factor(movieId), userId = as.factor(userId), month = as.factor(month), year = as.factor(year), hour = as.factor(hour))

summary(edx)

edx %>% arrange(desc(num_of_ratings)) %>% select(movieId, title, sum_of_ratings, num_of_ratings, avg_rating) %>% unique()

## Explore correlation between predictors

heatMapPalette <- colorRampPalette(rev(brewer.pal(11, "RdBu")))

# edx %>% ggplot(aes(hour, rating)) + geom_tile()

## Divide into train and test sets

set.seed(1, sample.kind = "Rounding")

test_index <- createDataPartition(edx$rating, times = 1, p = 0.7, list = FALSE)

test_set <- edx[test_index,]
train_set <- edx[-test_index,]

## Explore more relevant predictors

cor(train_set$sum_of_ratings, train_set$rating)

## Regularization

## Matrix factorization

## Train models

fit.lm1 <- train_set %>% train(rating ~ movieId + userId, data = ., method = "lm")

fit.rpart1 <- train_set %>% train(rating ~ movieId + userId + month + weekday + hour, data = ., method = "rpart")

## Test models

RMSE <- function(predicted, actual) {
  sqrt(mean((predicted - actual)^2))
}

pred.lm1 <- predict(fit.lm1, test_set)
RMSE(pred.lm1, test_set$rating)

pred.rpart1 <- predict(fit.rpart1, test_set)
RMSE(pred.rpart1, test_set$rating)

## Cross validate models

## Validate models