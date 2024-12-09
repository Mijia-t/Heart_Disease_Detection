import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

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
    
    return X, y, scaler

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

# Save user input data to a file
def save_user_input_to_file(input_data, name):
    '''
    Save user input data to a CSV file based on their name.
    
    If the file for the individul already exists, add the data;
    
    If not create a new one file and add the data.
    
    Parameters
    ----------
    input_data : list
        User's input data.
    name : str
        The user's name, used to generate the file name.
    '''
    user_filename = f"{name}_data.csv"
    
    # Create a DataFrame from the input data
    input_data_df = pd.DataFrame([input_data], columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                                                       'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                                                       'Oldpeak', 'ST_Slope'])
    
    # Check if the file already exists
    if os.path.exists(user_filename):
        input_data_df.to_csv(user_filename, mode='a', header=False, index=False)
    else:
        # If file does not exist, create it and write the header
        input_data_df.to_csv(user_filename, mode='w', header=True, index=False)
    
    print(f"User data saved to {user_filename}.")


# Function for user input and prediction
def input_data_for_prediction(model, scaler):
    '''
    Collect user input and predict heart disease.

    Parameters
    ----------
    model : object
        Trained model to make predictions.
    scaler : object
        Scaler used to normalize the input data.
    '''
    print("Please enter the data:")
    
    # Get user input (first ask for name)
    name = input("Please enter your name: ")
    
    # Get user input data
    age = float(input("Age: "))
    sex = input("Sex (M/F): ").upper()
    chest_pain_type = input("Chest Pain Type (TA/ATA/NAP/ASY): ").upper()
    resting_bp = float(input("Resting Blood Pressure: "))
    cholesterol = float(input("Cholesterol: "))
    fasting_bs = int(input("Fasting Blood Sugar (1: if > 120 mg/dl, 0: otherwise): "))
    resting_ecg = input("Resting ECG (Normal/ST/LVH): ").upper()
    max_hr = float(input("Max Heart Rate: "))
    exercise_angina = input("Exercise Angina (Y/N): ").upper()
    oldpeak = float(input("Oldpeak: "))
    st_slope = input("ST Slope (Up/Flat/Down): ").upper()
    
    # Prepare the data for prediction
    input_data = [age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg,
                   max_hr, exercise_angina, oldpeak, st_slope]
    
    input_data_df = pd.DataFrame([input_data], columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                                                     'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                                                     'Oldpeak', 'ST_Slope'])
    
    # Save user input to the respective user's file
    save_user_input_to_file(input_data, name)
    
    # Label encoding
    label_encoder = LabelEncoder()
    input_data_df['Sex'] = label_encoder.fit_transform(input_data_df['Sex'])
    input_data_df['ChestPainType'] = label_encoder.fit_transform(input_data_df['ChestPainType'])
    input_data_df['RestingECG'] = label_encoder.fit_transform(input_data_df['RestingECG'])
    input_data_df['ExerciseAngina'] = label_encoder.fit_transform(input_data_df['ExerciseAngina'])
    input_data_df['ST_Slope'] = label_encoder.fit_transform(input_data_df['ST_Slope'])
    
    # Standardize the input data
    input_data_scaled = scaler.transform(input_data_df)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    
    # Output the prediction
    if prediction[0] == 1:
        print("Prediction: Heart Disease detected!")
        print("Recommendation: Strongly advise to consult a doctor immediately for further examination and possible treatment.")
    else:
        print("Prediction: No Heart Disease detected.")
        print("Recommendation: Continue monitoring the health and maintain a healthy lifestyle.")

# View user health history
def view_user_health_history(name):
    '''
    View the health history of the individual by name.

    Parameters
    ----------
    name : str
        The user's name, used to locate the health history file.
    
    Returns
    -------
    None
    '''
    
    user_filename = f"{name}_data.csv"
    if os.path.exists(user_filename):
        df = pd.read_csv(user_filename)
        print(f"\nHealth history for {name}:")
        print(df)
    else:
        print(f"No records found for {name}.")
        
# Run the model
if __name__ == "__main__":
    file_path = "heart.csv"
    
    # Load the trained model and scaler for prediction
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Get user input and predict
    input_data_for_prediction(model, scaler)
    
    # View user's health history
    user_name = input("Enter your name to view your health history: ")
    view_user_health_history(user_name)


