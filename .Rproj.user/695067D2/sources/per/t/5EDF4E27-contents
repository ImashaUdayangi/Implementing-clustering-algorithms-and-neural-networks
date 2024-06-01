# Load required libraries
library(neuralnet)
library(readxl)
library(dplyr)
library(Metrics)
library(ggplot2)

# Read the Excel file
currency_data <- read_excel("ExchangeUSD.xlsx")

# Extract the relevant column (USD/EUR exchange rate)
exchange_rates <- as.data.frame(currency_data[3])
exchange_rates
str(exchange_rates)
summary(exchange_rates)

# Get the minimum and maximum values of the original data
data_min <- min(exchange_rates)
data_max <- max(exchange_rates)

# Function to Normalize the data
normalize_data <- function(data) {
  return ((data - min(data)) / (max(data) - min(data)))
}

# Function to de-normalize the data
denormalize_data <- function(normalized_value, min_value, max_value) {
  return ((max_value - min_value) * normalized_value + min_value)
}


# Create lagged variables
lag_1 <- lag(exchange_rates, 1)
lag_2 <- lag(exchange_rates, 2)
lag_3 <- lag(exchange_rates, 3)
lag_4 <- lag(exchange_rates, 4)

# Combine the original and lagged variables into different datasets(I/O Matrix)
dataset_1 <- cbind(exchange_rates, lag_1)
dataset_2 <- cbind(dataset_1, lag_2)
dataset_3 <- cbind(dataset_2, lag_3)
dataset_4 <- cbind(dataset_3, lag_4)

# Remove rows with missing values
dataset_1 <- na.omit(dataset_1)
dataset_2 <- na.omit(dataset_2)
dataset_3 <- na.omit(dataset_3)
dataset_4 <- na.omit(dataset_4)

# Rename columns
colnames(dataset_1) <- c('target', 'input_1')
colnames(dataset_2) <- c('target', 'input_1', 'input_2')
colnames(dataset_3) <- c('target', 'input_1', 'input_2', 'input_3')
colnames(dataset_4) <- c('target', 'input_1', 'input_2', 'input_3', 'input_4')


#Normalizing the data
normalized_data_1 <- as.data.frame(lapply(dataset_1, normalize_data))
normalized_data_2 <- as.data.frame(lapply(dataset_2, normalize_data))
normalized_data_3 <- as.data.frame(lapply(dataset_3, normalize_data))
normalized_data_4 <- as.data.frame(lapply(dataset_4, normalize_data))

# Create a box plot of the normalized data
boxplot(normalized_data_1)
boxplot(normalized_data_2)
boxplot(normalized_data_3)
boxplot(normalized_data_4)

# Split data into training and testing sets
train_set_1 <- normalized_data_1[1:400, ]
test_set_1 <- normalized_data_1[401:nrow(normalized_data_1), ]

train_set_2 <- normalized_data_2[1:400, ]
test_set_2 <- normalized_data_2[401:nrow(normalized_data_2), ]

train_set_3 <- normalized_data_3[1:400, ]
test_set_3 <- normalized_data_3[401:nrow(normalized_data_3), ]

train_set_4 <- normalized_data_4[1:400, ]
test_set_4 <- normalized_data_4[401:nrow(normalized_data_4), ]

# Set seed for reproducibility
set.seed(123)

# Define neural network models
# 1 hidden layer with 5 nodes from t-1 to t-4 
nn_model_1 <- neuralnet(target ~ input_1, data = train_set_1, hidden = c(5), linear.output = TRUE)
nn_model_2 <- neuralnet(target ~ input_1 + input_2, data = train_set_2, hidden = c(5), linear.output = TRUE)
nn_model_3 <- neuralnet(target ~ input_1 + input_2 + input_3, data = train_set_3, hidden = c(5), linear.output = TRUE)
nn_model_4 <- neuralnet(target ~ input_1 + input_2 + input_3 + input_4, data = train_set_4, hidden = c(5), linear.output = TRUE)

# 1 hidden layer with 10 nodes and 2 hidden layers with 5 and 3 nodes from t-1 to t-4
nn_model_5 <- neuralnet(target ~ input_1, data = train_set_1, hidden = c(10), linear.output = TRUE)
nn_model_6 <- neuralnet(target ~ input_1, data = train_set_1, hidden = c(8, 3), linear.output = TRUE)
nn_model_7 <- neuralnet(target ~ input_1 + input_2, data = train_set_2, hidden = c(10), linear.output = TRUE)
nn_model_8 <- neuralnet(target ~ input_1 + input_2, data = train_set_2, hidden = c(8, 3), linear.output = TRUE)
nn_model_9 <- neuralnet(target ~ input_1 + input_2 + input_3, data = train_set_3, hidden = c(10), linear.output = TRUE)
nn_model_10 <- neuralnet(target ~ input_1 + input_2 + input_3, data = train_set_3, hidden = c(8, 3), linear.output = TRUE)
nn_model_11 <- neuralnet(target ~ input_1 + input_2 + input_3 + input_4, data = train_set_4, hidden = c(10), linear.output = TRUE)
nn_model_12 <- neuralnet(target ~ input_1 + input_2 + input_3 + input_4, data = train_set_4, hidden = c(8, 3), linear.output = TRUE)

#plot neural network models
plot(nn_model_1)
plot(nn_model_2)
plot(nn_model_3)
plot(nn_model_4)
plot(nn_model_5)
plot(nn_model_6)
plot(nn_model_7)
plot(nn_model_8)
plot(nn_model_9)
plot(nn_model_10)
plot(nn_model_11)
plot(nn_model_12)

#De-normalize the Actual values for testing sets
actual_test_1 <- denormalize_data(test_set_1$target, data_min, data_max)
actual_test_2 <- denormalize_data(test_set_2$target, data_min, data_max)
actual_test_3 <- denormalize_data(test_set_3$target, data_min, data_max)
actual_test_4 <- denormalize_data(test_set_4$target, data_min, data_max)

#De-normalize the Predicted values for testing sets
predicted_test_1 <- denormalize_data(predict(nn_model_1, test_set_1), data_min, data_max)
predicted_test_2 <- denormalize_data(predict(nn_model_2, test_set_2), data_min, data_max)
predicted_test_3 <- denormalize_data(predict(nn_model_3, test_set_3), data_min, data_max)
predicted_test_4 <- denormalize_data(predict(nn_model_4, test_set_4), data_min, data_max)
predicted_test_5 <- denormalize_data(predict(nn_model_5, test_set_1), data_min, data_max)
predicted_test_6 <- denormalize_data(predict(nn_model_6, test_set_1), data_min, data_max)
predicted_test_7 <- denormalize_data(predict(nn_model_7, test_set_2), data_min, data_max)
predicted_test_8 <- denormalize_data(predict(nn_model_8, test_set_2), data_min, data_max)
predicted_test_9 <- denormalize_data(predict(nn_model_9, test_set_3), data_min, data_max)
predicted_test_10 <- denormalize_data(predict(nn_model_10, test_set_3), data_min, data_max)
predicted_test_11 <- denormalize_data(predict(nn_model_11, test_set_4), data_min, data_max)
predicted_test_12 <- denormalize_data(predict(nn_model_12, test_set_4), data_min, data_max)

# Function to calculate RMSE
calculate_rmse <- function(actual, predicted) {
  sqrt(mean((predicted - actual)^2))
}

# Function to calculate MAE
calculate_mae <- function(actual, predicted) {
  mean(abs(predicted - actual))
}

# Function to calculate MAPE
calculate_mape <- function(actual, predicted) {
  mean(abs((predicted - actual) / actual)) * 100
}

# Function to calculate sMAPE
calculate_smape <- function(actual, predicted) {
  2 * mean(abs(predicted - actual) / (abs(actual) + abs(predicted))) * 100
}

# Calculate evaluation metrics for each model
model_metrics <- data.frame(
  Model = 1:12,
  RMSE = c(
    calculate_rmse(actual_test_1, predicted_test_1),
    calculate_rmse(actual_test_2, predicted_test_2),
    calculate_rmse(actual_test_3, predicted_test_3),
    calculate_rmse(actual_test_4, predicted_test_4),
    calculate_rmse(actual_test_1, predicted_test_5),
    calculate_rmse(actual_test_1, predicted_test_6),
    calculate_rmse(actual_test_2, predicted_test_7),
    calculate_rmse(actual_test_2, predicted_test_8),
    calculate_rmse(actual_test_3, predicted_test_9),
    calculate_rmse(actual_test_3, predicted_test_10),
    calculate_rmse(actual_test_4, predicted_test_11),
    calculate_rmse(actual_test_4, predicted_test_12)
  ),
  MAE = c(
    calculate_mae(actual_test_1, predicted_test_1),
    calculate_mae(actual_test_2, predicted_test_2),
    calculate_mae(actual_test_3, predicted_test_3),
    calculate_mae(actual_test_4, predicted_test_4),
    calculate_mae(actual_test_1, predicted_test_5),
    calculate_mae(actual_test_1, predicted_test_6),
    calculate_mae(actual_test_2, predicted_test_7),
    calculate_mae(actual_test_2, predicted_test_8),
    calculate_mae(actual_test_3, predicted_test_9),
    calculate_mae(actual_test_3, predicted_test_10),
    calculate_mae(actual_test_4, predicted_test_11),
    calculate_mae(actual_test_4, predicted_test_12)
  ),
  MAPE = c(
    calculate_mape(actual_test_1, predicted_test_1),
    calculate_mape(actual_test_2, predicted_test_2),
    calculate_mape(actual_test_3, predicted_test_3),
    calculate_mape(actual_test_4, predicted_test_4),
    calculate_mape(actual_test_1, predicted_test_5),
    calculate_mape(actual_test_1, predicted_test_6),
    calculate_mape(actual_test_2, predicted_test_7),
    calculate_mape(actual_test_2, predicted_test_8),
    calculate_mape(actual_test_3, predicted_test_9),
    calculate_mape(actual_test_3, predicted_test_10),
    calculate_mape(actual_test_4, predicted_test_11),
    calculate_mape(actual_test_4, predicted_test_12)
  ),
  sMAPE = c(
    calculate_smape(actual_test_1, predicted_test_1),
    calculate_smape(actual_test_2, predicted_test_2),
    calculate_smape(actual_test_3, predicted_test_3),
    calculate_smape(actual_test_4, predicted_test_4),
    calculate_smape(actual_test_1, predicted_test_5),
    calculate_smape(actual_test_1, predicted_test_6),
    calculate_smape(actual_test_2, predicted_test_7),
    calculate_smape(actual_test_2, predicted_test_8),
    calculate_smape(actual_test_3, predicted_test_9),
    calculate_smape(actual_test_3, predicted_test_10),
    calculate_smape(actual_test_4, predicted_test_11),
    calculate_smape(actual_test_4, predicted_test_12)
  ),
  Structure = c(
    "No of Inputs: 1, No of Hidden Layers: 1, No of nodes: 5",
    "No of Inputs: 2, No of Hidden Layers: 1, No of nodes: 5",
    "No of Inputs: 3, No of Hidden Layers: 1, No of nodes: 5",
    "No of Inputs: 4, No of Hidden Layers: 1, No of nodes: 5",
    "No of Inputs: 1, No of Hidden Layers: 1, No of nodes: 10",
    "No of Inputs: 1, No of Hidden Layers: 2, No of nodes: 8,3",
    "No of Inputs: 2, No of Hidden Layers: 1, No of nodes: 10",
    "No of Inputs: 2, No of Hidden Layers: 2, No of nodes: 8,3",
    "No of Inputs: 3, No of Hidden Layers: 1, No of nodes: 10",
    "No of Inputs: 3, No of Hidden Layers: 2, No of nodes: 8,3",
    "No of Inputs: 4, No of Hidden Layers: 1, No of nodes: 10",
    "No of Inputs: 4, No of Hidden Layers: 2, No of nodes: 8,3"
  )
)

# Print the model evaluation metrics
print(model_metrics)

# Visualize the best performing model
best_model_index <- which.min(model_metrics$RMSE)
best_model_data <- data.frame(
  Actual = actual_test_2,
  Predicted = predicted_test_7
)

ggplot(best_model_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(
    x = "Actual Exchange Rate",
    y = "Predicted Exchange Rate",
    title = "Actual vs. Predicted Exchange Rates (Best Model)"
  ) +
  theme_minimal()
