library(e1071)

# read data.csv
data <- as.data.frame( read.csv("data.csv") )

# change attribute name to integer
# machine class -> [1, 2, 3]
machines <- 1:3
names(machines) <- c("Local", "Cluster", "Cloud")
data$class <- as.numeric(machines[as.vector(data$class)])

# normalize attributes
#data.norm <- data
#for (column in c("size", "population", "generation", "time", "class")) {
#	x <- data[, column]
#	nx <- (x - mean(x)) / sd(x)
#	data.norm[, column] <- nx
#}

# separate dataset into train and test sets
n <- nrow(data)
index <- 1:n
train.index <- sample(index, size=trunc(0.75*n))

train.set <- data[train.index,]
test.set <- data[-train.index,]

# separate train set into X (input data) and y (output data)
X.train <- subset(train.set, select=c("size", "population", "generation", "class"))
y.train <- subset(train.set, select=c("time"))

# separate test set into X (input data) and y (output data)
X.test <- subset(test.set, select=c("size", "population", "generation", "class"))
y.test <- subset(test.set, select=c("time"))

# tune parameters for SVM
tuneResult <- tune(svm, X.train, y.train, ranges=list(epsilon=seq(0, 1, 0.1), cost=2^(2:9)))

# plot parameters VS error (color)
plot(tuneResult)

# print best parameters
print(tuneResult)

# Root Mean Squared Error
rmse <- function(error) {
	sqrt(mean(error^2))
}

# use the best model from tuning to predict the output for the test input data
pred <- predict(tuneResult$best.model, X.test)

# compute the RMSE for the predicted output VS. expected output
print(rmse(pred - y.test))

