import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


def evaluate_models(X_test, y_test, model_dir):
    # Load models
    heuristic_model = joblib.load(f'{model_dir}/heuristic.joblib')
    baseline_model1 = joblib.load(f'{model_dir}/baseline_model1.joblib')
    baseline_model2 = joblib.load(f'{model_dir}/baseline_model2.joblib')
    nn_model = tf.keras.models.load_model(f'{model_dir}/nn_model')

    # Evaluate heuristic model
    heuristic_pred = heuristic_model.predict(X_test)
    heuristic_acc = accuracy_score(y_test, heuristic_pred)
    print('Heuristic model:')
    print(f'Accuracy: {heuristic_acc:.4f}')
    print(classification_report(y_test, heuristic_pred))

    # Evaluate baseline model 1
    baseline_pred1 = baseline_model1.predict(X_test)
    baseline_acc1 = accuracy_score(y_test, baseline_pred1)
    print('Baseline model 1:')
    print(f'Accuracy: {baseline_acc1:.4f}')
    print(classification_report(y_test, baseline_pred1))

    # Evaluate baseline model 2
    baseline_pred2 = baseline_model2.predict(X_test)
    baseline_acc2 = accuracy_score(y_test, baseline_pred2)
    print('Baseline model 2:')
    print(f'Accuracy: {baseline_acc2:.4f}')
    print(classification_report(y_test, baseline_pred2))

    # Evaluate neural network
    nn_pred = np.argmax(nn_model.predict(X_test), axis=-1)
    nn_acc = accuracy_score(y_test, nn_pred)
    print('Neural network model:')
    print(f'Accuracy: {nn_acc:.4f}')
    print(classification_report(y_test, nn_pred))
