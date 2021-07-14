
# Setting up

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(GGally)
library(ggridges)
library(ggthemes)

sales <- fread(text = gsub("::", "\t", readLines(unzip("archive.zip", "nyc-rolling-sales.csv"))), col.names = c("num", "borough", "neighborhood", "buildingClass"))

