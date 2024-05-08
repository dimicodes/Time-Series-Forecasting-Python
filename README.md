# Time Series Forecasting for Transaction Volumes

## Project Overview
This project utilizes several statistical and machine learning models to forecast transaction volumes based on historical data. Models employed include ARIMA, Prophet, Neural Networks (NN), Long Short-Term Memory Networks (LSTM), and Light Gradient Boosting Machine (LightGBM). The project aims to compare these models based on their predictive accuracy and to identify the most effective approach for time series forecasting in transaction data.

## Models Used
- **ARIMA (AutoRegressive Integrated Moving Average)**: Useful for understanding and predicting future points in a series.
- **Prophet**: Developed by Facebook, excellent for datasets with strong seasonal effects and several seasons of historical data.
- **Neural Networks**: Capable of modeling more complex relationships in data.
- **LSTM (Long Short-Term Memory)**: Effective for predictions based on time series data where the sequence of data is important.
- **LightGBM**: A gradient boosting framework that uses tree-based learning algorithms.

## Setup
### Prerequisites
- Python 3.8+
- Pandas, NumPy, Matplotlib, Scikit-Learn, TensorFlow, LightGBM, Prophet, Statsmodels

### Installation
1. Clone the repository:
```git clone https://github.com/dimicodes/forecasting-project.git```
2. Navigate to the project directory:
```cd forecasting-project```
3. Install the required packages:
```pip install -r requirements.txt```


## Usage
To run the forecasting models and evaluate their performance:
1. Ensure the dataset is located at `datasets/forecasting_data.csv`.
2. Run the main script:
```python main.py```
3. Review the output predictions and model evaluations which will be printed in the console.

## Data
The dataset used in this project contains daily transaction data spanning several years. Each entry records the date and the number of transactions that occurred on that date.

## Preview

![image](https://github.com/dimicodes/Time-Series-Forecasting-Python/assets/45632694/27065e64-445a-4ac9-b3f9-36323ddc4287)



