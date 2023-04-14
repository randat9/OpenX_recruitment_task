import pandas as pd

# Load the dataset
df = pd.read_csv('covtype.data', header=None)

# Define the heuristic function to classify the data
def heuristic_classify(row):
    # The heuristic simply looks at the Elevation feature to determine the class
    if row['Elevation'] <= 2800:
        return 1  # Class 1 (Spruce/Fir)
    else:
        return 2  # Class 2 (Not Spruce/Fir)

# Apply the heuristic to each row and save the results in a new column
df['heuristic_class'] = df.apply(heuristic_classify, axis=1)

# Save the results to a CSV file
df.to_csv('covtype_with_heuristic.csv', index=False)
