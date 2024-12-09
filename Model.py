import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data from a CSV file dynamically
def load_data(file_path):
    '''
    Load data from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    Returns
    -------
    object
        Loaded data as a pandas DataFrame.
    '''
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        raise FileNotFoundError(f"File {file_path} does not exist.")

# Data preprocessing
def preprocess_data(data):
    '''
    Preprocess the data by encoding categorical features and normalizing the features.

    Parameters
    ----------
    data : object
        The input data from csv.

    Returns
    -------
    X : array
        Normalized feature data.
    y : array
        Target variable.
    '''
    label_encoder = LabelEncoder()
    # Handle categorical features (includes Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope)
    data['Sex'] = label_encoder.fit_transform(data['Sex'])
    data['ChestPainType'] = label_encoder.fit_transform(data['ChestPainType'])
    data['RestingECG'] = label_encoder.fit_transform(data['RestingECG'])
    data['ExerciseAngina'] = label_encoder.fit_transform(data['ExerciseAngina'])
    data['ST_Slope'] = label_encoder.fit_transform(data['ST_Slope'])
    
    # Split into features and target label
    X = data.drop('HeartDisease', axis=1)  # Features
    y = data['HeartDisease']  # Target label
    
    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

# Split the data into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):
    '''
    Split the data into training and testing sets.

    Parameters
    ----------
    X : array
        Feature data.
    y : array
        Target variable.
    test_size : float
        The proportion of data to be used for testing.
    random_state : int
        The seed for random number generation.

    Returns
    -------
    tuple
        Tuple containing the training and testing data (X_train, X_test, y_train, y_test).
    '''
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train the Random Forest model
def train_random_forest(X_train, y_train):
    '''
    Train a Random Forest model.

    Parameters
    ----------
    X_train : array
        Feature data for training.
    y_train : array
        Target variable for training.

    Returns
    -------
    object
        Trained Random Forest model.
    '''
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model performance
def evaluate_model(model, X_test, y_test):
    '''
    Evaluate the model and get the accuracy.

    Parameters
    ----------
    model : object
        The trained model.
    X_test : array
        Feature data for testing.
    y_test : array
        Target variable for testing.

    Returns
    -------
    float
        The accuracy of the model.
    '''
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Main function to run the entire model
def run_model(file_path):
    '''
    Run the entire model: load data, preprocess, split, train, and evaluate.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    '''
    # Load data
    data = load_data(file_path)
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train the model
    model = train_random_forest(X_train, y_train)
    
    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    
    print(f"Model accuracy: {accuracy * 100:.2f}%")

# Run the model
if __name__ == "__main__":
    file_path = "heart.csv"
    run_model(file_path)










