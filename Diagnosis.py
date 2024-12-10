import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

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

def save_user_input_to_file(input_data, name, date_input):
    '''
    Save user input data to a CSV file based on their name, and include the date.

    Parameters
    ----------
    input_data : list
        User's input data.
    name : str
        The user's name, used to generate the file name.
    date_input : str
        The date provided by the user.
    '''
    user_filename = f"{name}_data.csv"
    
    # Use the provided date
    input_data.append(date_input)
    
    # Create a DataFrame from the input data
    columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
               'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
               'Oldpeak', 'ST_Slope', 'Date']
    
    input_data_df = pd.DataFrame([input_data], columns=columns)
    
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
    
    # Get the date from the user
    date_input = input("Enter the date (YYYY-MM-DD): ")
    
    # Prepare the data for prediction
    input_data = [age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg,
                   max_hr, exercise_angina, oldpeak, st_slope]
    
    input_data_df = pd.DataFrame([input_data], columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                                                     'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                                                     'Oldpeak', 'ST_Slope'])
    
    # Check if there is extreme abnormal value
    abnormal_data = check_extreme_abnormal_data(resting_bp, cholesterol, max_hr)
    if abnormal_data:
        print(f"Extreme abnormal data detected: {', '.join(abnormal_data)}")
    else:
        print('No extreme abnormal data detected')
        
    # Risk detection
    risk_data = risk_detection(resting_bp, cholesterol, max_hr)
    print(f"Risk Levels: {risk_data}")
 
    
    # Save user input to the respective user's file
    save_user_input_to_file(input_data, name, date_input)
    
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


# Check extreme abonormal value
def check_extreme_abnormal_data(resting_bp, cholesterol, max_hr):
    '''
    Check if there is abnormal value

    Parameters
    ----------
    resting_bp : float
        Resting blood pressure.
    cholesterol : float
        Cholesterol level.
    max_hr : float
        Max heart rate.

    Returns
    -------
    abnormal_data : list
        List of abnormal data.
    '''
    abnormal_data = []
    
    # Check if RestingBP is in normal range: 60 - 200
    if resting_bp < 60 or resting_bp > 200:
        abnormal_data.append(f"RestingBP: {resting_bp}")
    
    # Check if cholesterol level is in normal range : 40 - 500
    if cholesterol < 40 or cholesterol > 500:
        abnormal_data.append(f"Cholesterol: {cholesterol}")
    
    # Check if Max heart rate is in  normal range
    if max_hr < 40 or max_hr > 140:
        abnormal_data.append(f"MaxHR: {max_hr}")
    
    return abnormal_data

def risk_detection(resting_bp, cholesterol, max_hr):
    '''
    Risk detection based on the user input of resting BP, cholesterol, and max heart rate.

    Parameters
    ----------
    resting_bp : float
        Resting blood pressure (single value).
    cholesterol : float
        Cholesterol level.
    max_hr : float
        Max heart rate.

    Returns
    -------
    risk_data : dict
        Risk levels for each input parameter: {'RestingBP': risk_level, 'Cholesterol': risk_level, 'MaxHR': risk_level}
    '''
    risk_data = {}

    # Risk detection for RestingBP
    if 60 < resting_bp < 120:
        risk_data['RestingBP'] = 'normal'
    elif 120 <= resting_bp < 130:
        risk_data['RestingBP'] = 'risk1'
    elif 130 <= resting_bp < 140:
        risk_data['RestingBP'] = 'risk2' 
    else:
        risk_data['RestingBP'] = 'risk3' 

    # Risk detection for Cholesterol
    if 40 < cholesterol < 200:
        risk_data['Cholesterol'] = 'normal'
    elif 200 <= cholesterol < 240:
        risk_data['Cholesterol'] = 'risk1' 
    elif 240 <= cholesterol < 500:
        risk_data['Cholesterol'] = 'risk2'
    else:
        risk_data['Cholesterol'] = 'risk3'

    # Risk detection for MaxHR
    if 60 <= max_hr <= 100:
        risk_data['MaxHR'] = 'normal'
    elif 101 <= max_hr < 120:  
        risk_data['MaxHR'] = 'risk1'
    elif 121 <= max_hr < 140:  
        risk_data['MaxHR'] = 'risk2'
    else:
        risk_data['MaxHR'] = 'risk3'  
    
    return risk_data
      
# View user health history
def view_user_health_history(name):
    '''
    View the health history of the individual by name and perform trend analysis.

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

# View visualization file
def view_visualization_file(name):
    '''
    View the saved health trend visualization file.

    Parameters
    ----------
    name : str
        The user's name, used to locate the saved health trend file.

    Returns
    -------
    None
    '''
    plot_filename = f"{name}_health_trend.png"
    
    # Check if the plot already exists
    if os.path.exists(plot_filename):
        print(f"Displaying saved visualization for {name}:")
        img = plt.imread(plot_filename)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print(f"No saved visualization found for {name}. Generating a new one...")
        
        # Generate the health trend plot from the CSV file
        user_filename = f"{name}_data.csv"
        if os.path.exists(user_filename):
            df = pd.read_csv(user_filename)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            plt.figure(figsize=(10, 6))
            df['RestingBP'].plot(label='Resting Blood Pressure', color='green', title=f"Health Trend of {name}")
            df['Cholesterol'].plot(label='Cholesterol', color='blue')
            df['MaxHR'].plot(label='Max Heart Rate', color='orange') 
            plt.xlabel('Date')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)
            
            # Save the new plot as a PNG file
            plt.savefig(plot_filename)
            print(f"Health trend plot saved to {plot_filename}")
            plt.show()
        else:
            print(f"No health data found for {name}. Cannot generate visualization.")
            
def view_diagnosis_history(name):
    '''
    View the diagnosis history for a user
    
    Parameters
    ----------
    name : str
        The user's name.

    Returns
    -------
    None
    '''
    user_filename = f"{name}_data.csv"
    output_filename = f"{name}_diagnosis_history.csv"
    
    if os.path.exists(user_filename):
        df = pd.read_csv(user_filename)
        print(f"\nDiagnosis history for {name}:")
        
        # List to store history data for saving to a file
        history_data = []

        for index, row in df.iterrows():
            abnormal_data = check_extreme_abnormal_data(row['RestingBP'], row['Cholesterol'], row['MaxHR'])
            risk_data = risk_detection(row['RestingBP'], row['Cholesterol'], row['MaxHR'])

            record = {
                "Date": row['Date'],
                "ExtremeAbnormalData": ', '.join(abnormal_data) if abnormal_data else 'None',
                "RiskLevels": risk_data   
            }
            history_data.append(record)
            
            print(f"{row['Date']}; Extreme abnormal data: {record['ExtremeAbnormalData']}; "
                  f"Risk Levels: {record['RiskLevels']}")
        
        # Save to CSV
        history_df = pd.DataFrame(history_data)
        history_df.to_csv(output_filename, index=False)
        print(f"Diagnosis history saved to {output_filename}.")
    else:
        print(f"No diagnosis history found for {name}.")


# Main menu for selecting operations
def main_menu():
    '''
    Display a menu of options for the user to select one of the following operations:
    1. Input new data and make predictions.
    2. View the health history of the user from a CSV file.
    3. View or generate the saved visualization file of the user.
    4. View diagnosis history of the user from a CSV file.

    Based on the user's choice, the corresponding function is called to perform the operation.

    Returns
    -------
    None
    '''
    print("Please select an option:")
    print("1. Input new data")
    print("2. View user health history")
    print("3. View visualization file")
    print("4. View diagnosis history")
    
    choice = input("Enter your choice: ")
    
    if choice == '1':
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        input_data_for_prediction(model, scaler)
    elif choice == '2':
        user_name = input("Enter the user's name to view their health history: ")
        view_user_health_history(user_name)
    elif choice == '3':
        user_name = input("Enter the user's name to view visualization file: ")
        view_visualization_file(user_name)
    elif choice == '4':
        user_name = input("Enter the user's name to view diagnosis history: ")
        view_diagnosis_history(user_name)
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()


