# OpenX_recruitment_task

# Overview

This project aims to classify different forest cover types using machine learning algorithms and a neural network. The Covertype Data Set, sourced from the UCI Machine Learning Repository, is used for training and evaluation. The project includes the following steps:


1. Data loading and preprocessing: The dataset is loaded and transformed to prepare it for machine learning models.
2. Simple heuristic: A simple heuristic is implemented as a baseline classifier for the dataset.
3. Baseline models: Two simple machine learning models are trained and evaluated as additional baselines.
4. Neural network: A neural network is built and trained using TensorFlow to classify the cover types.
5. Hyperparameter tuning: A function is implemented to find the best set of hyperparameters for the neural network.
6. Model evaluation: The neural network and other models are evaluated using appropriate plots and metrics.
7. REST API: A simple REST API is created to serve the models.

# Requirements
The following libraries are required to run the code:

* pandas
* scikit-learn
* TensorFlow
* Flask
* joblib

# Files

* load_data.py: Loads and preprocesses the dataset.
* heuristic.py: Implements a simple heuristic for classifying the data.
* baseline_models.py: Trains and evaluates two baseline machine learning models.
* neural_network.py: Builds and trains the neural network model.
* evaluate_models.py: Evaluates the models using appropriate plots and metrics.
* api.py: Creates a REST API to serve the models.
* baseline_model_1.joblib: A saved instance of the first baseline model trained on the dataset.

# Running the Code

To run the code, simply clone the repository and run the api.py file. The REST API can be accessed by making requests to http://localhost:5000/predict. The API accepts the following parameters:

*model: The name of the model to use (heuristic, baseline1, baseline2, or neural_network).
*elevation: Elevation in meters (integer).
* aspect: Aspect in degrees azimuth (integer).
* slope: Slope in degrees (integer).
* horizontal_distance_to_hydrology: Horizontal distance to nearest surface water feature (integer).
* vertical_distance_to_hydrology: Vertical distance to nearest surface water feature (integer).
* horizontal_distance_to_roadways: Horizontal distance to nearest roadway (integer).
* hillshade_9am: Hillshade index at 9am, summer solstice (0 to 255 integer).
* hillshade_noon: Hillshade index at noon, summer solstice (0 to 255 integer).
* hillshade_3pm: Hillshade index at 3pm, summer solstice (0 to 255 integer).
* horizontal_distance_to_fire_points: Horizontal distance to nearest wildfire ignition points (integer).
* soil_type: Soil type (integer from 1 to 40).

Alternatively, you can run the individual Python files to perform specific tasks, such as training models or evaluating their performance.

# Conclusion

This project demonstrates the use of different machine learning models and a neural network to classify forest cover types using the Covertype Data Set. The results show that the neural network outperforms the other models, achieving an accuracy of over 85%. The project also includes a simple REST API and Docker containerization for easy deployment.
