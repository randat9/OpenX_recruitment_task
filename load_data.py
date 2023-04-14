import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2, random_state=42):
    # Load data into Pandas dataframe
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz', header=None)

    # Define feature names
    feature_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                     'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                     'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                     'Horizontal_Distance_To_Fire_Points'] + ['Wilderness_Area_{}'.format(i) for i in range(4)] + \
                    ['Soil_Type_{}'.format(i) for i in range(40)]

    # Set feature names as column names in dataframe
    df.columns = feature_names + ['Cover_Type']

    # Split dataframe into features and target
    X = df.drop('Cover_Type', axis=1)
    y = df['Cover_Type']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
