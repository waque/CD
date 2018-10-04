library(dplyr)
library(tidyverse)

# a)

algae <- read.csv("algae.csv")

# b)

algae %>% count()

# c) 

algae %>% summary()