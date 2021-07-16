library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(GGally)
library(ggridges)
library(ggthemes)

incomes <- fread(text = gsub("::", "\t", readLines(unzip("archive.zip", "adult.csv"))), col.names = c("age", "workclass", "fnlwgt", "education", "education.number", "marital.status", "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss", "hours.per.week", "native.country", "income"))

str(incomes)
dim(incomes)

colSums(incomes == "?")

incomes <- incomes %>% filter(!(workclass == "?" | occupation == "?" | native.country == "?"))

dim(incomes)

colSums(incomes == "?")

summary(incomes)

incomes <- incomes %>% mutate(workclass = as.factor(workclass), education = as.factor(education), marital.status = as.factor(marital.status), occupation = as.factor(occupation), relationship = as.factor(relationship), race = as.factor(race), sex = as.factor(sex), native.country = as.factor(native.country), income = as.factor(income))

summary(incomes)

ggcorr(incomes)

test_index <- createDataPartition(incomes$income, times = 1, p = 0.2, list = FALSE)

