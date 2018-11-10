
library(arules)
library(dplyr)
library(rCBA)

green <- read.csv('green_consensus.csv')
hin <- read.csv('hinselmann_consensus.csv')
sch <- read.csv('schiller_consensus.csv')

green <- green[, -which(names(green) %in% c("experts..0","experts..1", "experts..2", "experts..3", "experts..4", "consensus"))]
hin <-hin[, -which(names(green) %in% c("experts..0","experts..1", "experts..2", "experts..3", "experts..4", "consensus"))]
sch <- sch[, -which(names(green) %in% c("experts..0","experts..1", "experts..2", "experts..3", "experts..4", "consensus"))]

for (x in 1:(ncol(green))) {
    green[,x] <- discretize(green[,x],
                                breaks=3,
                                method="interval")
}



green <- as(green, "transactions")

rules <- fpgrowth(green, support = 0.05, confidence = 0.9, maxLength = 5, verbose = TRUE, parallel = TRUE)
inspect(rules)
