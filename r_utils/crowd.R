#!/usr/bin/Rscript
###GENERAL SETTINGS###
# Exploring the Crowdflower Data
library(readr)

train <- read_csv("../data/train.csv")
test  <- read_csv("../data/test.csv")

cat(names(train),"\n\n")

cat(unique(train$query)[1:10],"\n\n")

cat(length(setdiff(unique(train$query), unique(test$query))),"\n\n")
cat(unique(train$product_title)[1:10],"\n\n")

cat("The number of product titles that are only in the train set or only in the test set\n")
cat(length(setdiff(unique(train$product_title), unique(test$product_title))),"\n")

cat("The number of product titles that are in both the train and test sets\n")
cat(length(intersect(unique(train$product_title), unique(test$product_title))),"\n")


# We'll use the library ggvis for data visualization
library(ggvis)
# And the library tm to help with text processing
library(tm)