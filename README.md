# public-notebooks

Two projects are exposed here, both from the challenges of Kaggle.

## GiveMeSomeCredit

In this challenge, the objective is to provide a model to predict if a new customer will fail to pay his/her loan after two years, given his/her set of features. In the training set, there are 150,000 rows and 10 columns. The target variable is boolean, and it is heavily unbalanced. There are some missing values in the dataset.

I provide three notebooks:
- the exploratory data analysis of each feature;
- the tune of one model (the other models I used are the very same of this notebook);
- and the prediction using a stack of models.

## Instacart

> Instacart is an American company that operates grocery delivery service. Customers select groceries through a web application from various retailers and delivered by a personal shopper. (Wikipedia)

In this challenge, the objective is to recommend a set of products to the customers based on their past orders. In the training set, there are over 3 millions grocery orders, from more than 200,000 Instacart users. In each order, there are the set of bought products, the time of the day and the day of the week it happened, the number of days since last order, and other features.

I provide four notebooks:
- the exploratory data analysis concerning the products;
- the exploratory data analysis concerning the reordered products;
- the exploratory data analysis concerning the timing information of the orders;
- and a baseline prediction (I am working on a better predictor).
