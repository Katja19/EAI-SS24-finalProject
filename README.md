# FinalProject_EnterpriseAI
This template is a foundation for a possible final project. It includes a single pipeline, whose purpose is to create a machine-learning model that can predict the passenger frequency in Wuerzburg. If you want to work with this repository as a foundation of your project, you need to extend the repository in an extensive way.
**Your task consists of creating a pipeline that predicts the next 24 hours, starting from the last entry of the API return.** 

You can extend the dataset with other datasets to have more features. For example, you can use this weather API <a href="https://openweathermap.org/api"> HERE</a>. Also, you could use lagged features to extend the feature set. However, keep in mind that you must also fulfill the requirements stated in Wuecampus.

The data that is used in this repository is retrieved from the<a href="https://opendata.wuerzburg.de/explore/?refine.publisher=Stadt+Würzburg&sort=modified"> open data portal </a>, which provides different data sources about Wuerzburg.


## Pipeline
The pipeline consists of the following steps:
- update_data: It will fetch the data from the open data portal and save it to the SQLite database
- load_data: It will load the data from the SQLite database
- split_data: This step splits the dataframe into train and test as well as into input and output
- feature_engineering: The function uses a OnehotEncoder to transform the location column
- create_model: Creates a linear regression to predict the pedestrian count for the next hour based only on the location (leading to the mean of the training data)
- evaluate: Evaluate the created model by calculating the mean absolute value.
