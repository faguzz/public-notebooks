library(class)

x <- as.data.frame( read.csv("data.csv") )

# machine class -> numbers
machines <- 1:3
names(machines) <- c("Local", "Cluster", "Cloud")
x$class <- as.numeric(machines[as.vector(x$class)])

# TODO: normalize attributes

# train and test sets
n <- nrow(x)
index <- 1:n
train.index <- sample(index, size=trunc(0.75*n))
train.set <- x[train.index,]
test.set <- x[-train.index,]


# KNN -- selecting best k for time
res <- NULL
for (k in seq(1, 13, by=2)) {
	pred <- knn(train.set[,-c(4,5)], test.set[,-c(4,5)], train.set$time, k=k)
	pred <- as.double(as.vector(pred))
	mse <- mean((pred - test.set$time)^2)
	res <- rbind(res, c(k, mse))
}

k.time <- which.min(res[,2]) * 2 - 1

png("/tmp/knn_time.png")
plot(res, type='l', main="predicting time")
abline(v=res[k.time,1], lty=2)
dev.off()


# KNN -- selecting best k for fsize
res <- NULL
for (k in seq(1, 15, by=2)) {
	pred <- knn(train.set[,-c(4,5)], test.set[,-c(4,5)], train.set$filesize, k=k)
	pred <- as.double(as.vector(pred))
	mse <- mean((pred - test.set$filesize)^2)
	res <- rbind(res, c(k, mse))
}

k.fsize <- which.min(res[,2]) * 2 - 1

png("/tmp/knn_fsize.png")
plot(res, type='l', main="predicting filesize")
abline(v=res[k.fsize,1], lty=2)
dev.off()


# Experiment -- given parameters, compute time and fsize for every class of machine
for (i in 1:nrow(test.set)) {
	element <- test.set[i, -c(4,5)]

	res <- NULL
	for (j in machines) {
		row <- element
		row$class <- j

		pred.time <- knn(train.set[,-c(4,5)], row, train.set$time, k=k.time)
		pred.time <- as.double(as.vector(pred.time))
		pred.fsize <- knn(train.set[,-c(4,5)], row, train.set$filesize, k=k.fsize)
		pred.fsize <- as.double(as.vector(pred.fsize))

		row$pred.time <- pred.time
		row$pred.fsize <- pred.fsize
		res <- rbind(res, row)
	}
	print(res)

}

