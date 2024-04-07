# Crop Recommender Project


## Project Overview:
This project entails the development of a predictive model to recommend the most suitable crops to grow on a farm based on various parameters. The model is built from scratch and deployed using Streamlit, facilitating farmers in making data-driven decisions for crop selection. Additionally, a fastAPI endpoint is created to allow easy integration with other applications. 


## Parameters
The crop recommender system takes into account the following parameters:
- **N (Nitrogen):** Float value representing soil nitrogen content.
- **P (Phosphorus):** Float value representing soil phosphorus content.
- **K (Potassium):** Float value representing soil potassium content.
- **Temperature:** Float value representing the temperature of the farm area.
- **Humidity:** Float value representing the humidity of the farm area.
- **pH:** Float value representing the pH level of the soil.
- **Rainfall:** Float value representing the amount of rainfall in the farm area.

## Features:
- **Streamlit App:** A user-friendly interface is developed using Streamlit for easy access to the crop recommendation system. Farmers can input their soil and weather parameters to receive personalized crop recommendations.
- **Predictive Model:** The heart of the project is a predictive model trained on historical data to recommend crops based on the provided parameters. The model utilizes machine learning algorithms to analyze and predict the most suitable crops for the given conditions.
- **FastAPI Endpoint:** An API endpoint is created using FastAPI, enabling seamless integration of the crop recommendation system with other applications or services.


## Importance:
This project aims to empower farmers with data-driven insights for crop selection, ultimately improving crop yields and farm productivity. By leveraging machine learning techniques and deploying user-friendly interfaces, farmers can make informed decisions tailored to their specific farming conditions.

