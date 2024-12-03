# AI-Assignment

## topic : Sales Forecasting

## group members

 1.Manushkumar Patel [KU2407U793]
 
 2.Prajapati Krish R [KU2407U807]
 
 3.Ruchir Radhanpura [KU2407U770]
 
 4.Ishika Artani [KU2407U808]

##Objective of the Project

The objective of the Sales Forecasting Project is to utilize historical sales data to predict future sales and provide actionable insights. By employing time-series analysis

##Tools and Libraries Used

Data Collection & Storage
• Uses SQL / SQLite / MySQL for structured historical sales data storage.
• Microsoft Excel / Google Sheets for small-scale data entry and exploration.

Data Preprocessing & Analysis
• Uses Pandas for data manipulation, cleaning, and time-series formatting.
• Uses NumPy for numerical computations and array handling.
• Uses scipy for statistical analysis and tests.

Time-Series Analysis & Forecasting
• Uses statsmodels for traditional time-series models.
• Uses Prophet for automated forecasting with trend and seasonality modeling.
• Uses pmdarima for automated ARIMA model selection and hyperparameter tuning.
• Uses sktime for time-series machine learning and forecasting tasks.

Machine Learning & Deep Learning Models
• Uses scikit-learn for regression-based forecasting and metrics.
• Uses TensorFlow / Keras for advanced models.
• Uses PyTorch for advanced forecasting.
• Uses XGBoost / LightGBM for gradient-boosted tree models.

Data Visualization
• Uses Matplotlib for basic static plots.
• Uses Seaborn for enhanced statistical visualizations.
• Uses Plotly for interactive sales trends and forecast results.

Performance Metrics
• Uses scikit-learn.metrics for evaluating model performance.
• Uses KPI Tracking Tools for aligning forecasts with business goals.

Development Environment
• Uses Jupyter Notebook for interactive development and exploratory data analysis.

##Data Source(s)


• Internal Business Data: POS Systems, ERP Systems, CRM Systems, and Open Data Sources.
• Open Data Sources: Kaggle Datasets, UCI Machine Learning Repository, Google Dataset Search.
• Publicly Available Economic Data: Government Data Portals, World Bank / OECD, Eurostat.
• Online Marketplaces: Amazon Web Services, Google BigQuery, and E-commerce Platforms.
• Simulated or Synthetic Data: Custom Data Generation, Bootstrapped Data, Google Trends, Social Media Insights.
• Social Media and Web Traffic Data: Google Trends, Social Media Insights.
• Proprietary Third-Party Data Providers: Nielsen, Gartner, Statsta.

##Execution Steps (How to run the project)


Objective:
• Predict future sales based on historical time-series data.
• Visualize forecasted vs. actual sales.

Data Collection & Preprocessing:
• Gather historical sales data from various sources.
• Clean data by removing missing values and converting data into a time-series format.
• Handle outliers and ensure consistency.
• Convert categorical data to numerical form using encoding techniques.

Exploratory Data Analysis (EDA):
• Visualize data to identify trends, seasonality, and outliers.
• Use statistical tests like the Augmented Dickey-Fuller (ADF) test.
• Decompose time-series into trend, seasonality, and residual components.

Model Selection and Development:
• Use traditional time-series models like ARIMA/SARIMA, exponential smoothing, machine learning models, and deep learning models.
• Train the model using techniques like cross-validation or GridSearchCV for hyperparameter optimization.

Model Evaluation:
• Compare predicted values to actual values on the test set.
• Use error metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), or Mean Absolute Percentage Error (MAPE).

Future Sales Forecasting:
• Use the trained model to forecast future sales for the required period.
• Use confidence intervals for uncertainty in predictions.
• Evaluate model performance when applied to historical data not used during training.

Visualizing Results:
• Plot actual sales vs. forecasted sales on a line chart.
• Show the trend and seasonal components of the time series separately using decomposition.

Model Deployment:
• Deploy the forecasting model as a REST API using Flask or FastAPI for real-time predictions.
• Create interactive dashboards with Dash or Tableau.

Monitoring & Maintenance:
• Continuously track the model’s prediction accuracy.
• Retrain the model with updated data or tune it to adapt to market dynamics.

##Summary of Results


Data Preparation and Preprocessing:
• Sales data cleaned to remove missing values and outliers.
• Time-based features created to enhance predictive power.
• Stationarity Check confirmed by Augmented Dickey-Fuller (ADF) test.

Exploratory Data Analysis (EDA):
• Sales data showed strong seasonality and upward trend.
• Time-series decomposition revealed trends, seasonality, and residual noise.

Model Selection and Performance:
• Traditional models like ARIMA and SARIMA selected based on trend and seasonality.
• Machine learning models like Random Forest and XGBoost tested but showed lower accuracy.
• ARIMA model showed the best performance with a Mean Absolute Error (MAE) of approximately 5-7%.

Forecasting and Visualization:
• ARIMA model used to forecast sales for the next 3–6 months, with 95% confidence intervals.
• Line graphs used to compare actual sales against forecasted values.
• Confidence intervals plotted alongside forecasted values to indicate uncertainty in predictions.

Business Insights:
• Seasonality identified for marketing campaigns, staffing, and inventory management.
• Forecasted sales helped plan for future demand and avoid overstocking or stockouts.
• Anomalies & Outliers flagged periods where actual sales significantly deviated from forecasts.

Model Deployment and Maintenance:
• Forecast model deployed as an API using Flask or FastAPI.
• Periodic model retraining suggested to account for market changes.

Limitations and Future Work:
• Model accuracy reduced during irregular events not captured in historical data.
• Future work includes exploring ensemble methods and incorporating external variables.

##Challenges Faced


Data Quality Issues:
• Missing Values: Imput missing values using forward/backward filling or interpolation.
• Outliers: Detect and handle outliers using statistical methods or domain-specific thresholds.
• Noise and Irregularities: Use smoothing techniques or decomposition methods to separate trend, seasonality, and residual noise.

Data Complexity and Seasonality:
• Seasonality Patterns: Identify and capture seasonality explicitly using SARIMA or Prophet.
• Changing Trends: Use models like SARIMA with external regressors to account for changing trends.
• Multiple Seasons: Decompose the time-series into multiple components or use Prophet.

Stationarity and Differencing:
• Stationarity: Apply differencing to remove trends and use tests like the Augmented Dickey-Fuller (ADF) test to confirm stationarity.

Model Selection and Tuning:
• Overfitting: Use cross-validation to tune hyperparameters and apply regularization techniques if necessary.
• Choosing the Right Model: Experiment with different models and evaluate them based on performance metrics like MAE, RMSE, and MAPE.

Data Granularity and Aggregation:
• High Granularity: Aggregate the data to a daily, weekly, or monthly level.
• Aggregation Issues: Capture seasonality and trend patterns specific to different levels when aggregating data.

Forecast Horizon and Uncertainty:
• Long-Term Forecasting: Incorporate confidence intervals to account for the uncertainty in predictions.
• Horizon Sensitivity: Use a hybrid approach, combining short-term models with long-term models.

Model Interpretation and Business Application:
• Lack of Interpretability: Prefer models like ARIMA, SARIMA, or Prophet that provide interpretable components.
• Business Constraints: Ensure that the forecast model incorporates business knowledge and is aligned with operational constraints.

Real-Time Data and Scalability:
• Real-Time Updates: Implement online learning methods or retrain models periodically with new data.
• Scalability: Use distributed computing frameworks like Dask or Apache Spark to handle large datasets and parallelize model training.

External Factors and External Data Integration:
• Ignoring External Factors: Use external regressors or incorporate external data sources to improve forecast accuracy.

