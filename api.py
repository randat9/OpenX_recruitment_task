from flask import Flask, request, jsonify
import pickle
from heuristic import simple_heuristic
from baseline_models import simple_decision_tree, simple_logistic_regression
from neural_network import train_and_evaluate_model

app = Flask(__name__)

# Load saved models
with open('heuristic_model.pkl', 'rb') as f:
    heuristic_model = pickle.load(f)

with open('dt_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)

with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Define API routes
@app.route('/predict', methods=['POST'])
def predict():
    model = request.json['model']
    features = request.json['features']

    if model == 'heuristic':
        prediction = simple_heuristic(features)
    elif model == 'decision_tree':
        prediction = dt_model.predict([features])[0]
    elif model == 'logistic_regression':
        prediction = lr_model.predict([features])[0]
    elif model == 'neural_network':
        model_file = request.json['model_file']
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        prediction = model.predict([features])[0]
    else:
        return jsonify({'error': 'Invalid model choice'})

    return jsonify({'prediction': prediction})

@app.route('/train_nn', methods=['POST'])
def train_nn():
    # Get hyperparameters from request
    hidden_layers = request.json['hidden_layers']
    learning_rate = request.json['learning_rate']
    epochs = request.json['epochs']
    batch_size = request.json['batch_size']

    # Train and save model
    model_file = 'nn_model.pkl'
    train_and_evaluate_model(hidden_layers, learning_rate, epochs, batch_size, model_file)

    return jsonify({'message': 'Model trained successfully'})

if __name__ == '__main__':
    app.run(debug=True)

