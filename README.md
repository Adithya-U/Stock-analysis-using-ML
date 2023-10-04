# Stock Market Closing Price Prediction with Machine Learning Models

(Currently ongoing project)

## Introduction

This project aims to determine the best machine learning model for predicting the closing prices of 7000 NYSE stocks using time series data. We evaluate the performance of four different models: XGBoost, ARIMA, TBATS, and Recurrent Neural Networks (RNN) to find the most accurate and reliable approach for stock price prediction.

## Dependencies

To run this project, you'll need the following Python dependencies:

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Seaborn
- Scikit-Learn
- pandas_ta
- TA-Lib
- Boruta-py
- Statsmodels


## Dataset

The dataset used in this project consists of time series data for 7000 NYSE stocks. Each data point includes the date, stock identifier, and closing price. The dataset is available in the `data/` directory.

## Models Evaluated

We assess the following machine learning models for stock market closing price prediction:

1. **XGBoost**: A gradient boosting algorithm known for its excellent predictive performance.

2. **ARIMA (AutoRegressive Integrated Moving Average)**: A traditional time series forecasting method that models the relationship between data points and their lagged values.

3. **TBATS (Trigonometric Seasonal Decomposition of Time Series)**: A time series forecasting model capable of handling multiple seasonalities.

4. **Recurrent Neural Networks (RNN)**: A deep learning model designed to capture sequential patterns in time series data.

## Usage

1. Clone this repository to your local machine:

2. Navigate to the project directory:

3. Install the required Python libraries and dependencies:

4. Run the Jupyter Notebook or Python scripts to train and evaluate each model. 

5. Compare the performance metrics and analysis provided in the notebooks to determine the best model for your stock price prediction task.


After evaluating each model, we provide a detailed analysis of their performance based on various metrics such as Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and more. This analysis will help you make an informed decision about which model is most suitable for your stock market closing price prediction.

## Contributing

Contributions to this project are welcome! If you have suggestions, improvements, or would like to add more models for evaluation, please open an issue or create a pull request on GitHub.


## Acknowledgments

- The dataset used in this project contains stock market data from NYSE, and credit goes to the data providers.
- We appreciate the open-source community for developing and maintaining the machine learning libraries used in this project.

