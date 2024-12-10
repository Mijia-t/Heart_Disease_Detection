# Heart Disease Prediction and Management System
### Project Overview
This project is a comprehensive Heart Disease Prediction and Management System designed to help users monitor their heart health. By inputting personal health data, users can quickly assess their risk of heart disease. The system also provides functionalities for data storage, visualization, and detailed risk analysis. The modular design ensures independent yet interconnected functionality across various components.

### File Descriptions
The code of the previous files is kept in the order of the files, e.g. User_input_prediction.py is adding an additional new feature based on the Model.py. 
At the end of each file there is test as a unit test.

1. Model.py: Loads the dataset, trains the model, and generates predictions.
2. User_input_prediction.py: Allows users to input personal health data and get predictions.
3. User_file.py: Saves user inputs into a personalized CSV file.
4. View_history.py: Displays the user’s past health data.
5. Visualization.py: Visualizes user health data trends.
6. Main_tree_options.py: Provides a streamlined main menu with three independent options.
7. Diagnosis.py: Adds the ability to view diagnosis history and generate a detailed risk report.

### Input
The system accepts user-provided health data for prediction and analysis. 

Below are the input fields required:

Age: User's age (e.g., 18).

Sex: User's gender (M for Male, F for Female).

Chest Pain Type: Four options (TA, ATA, NAP, ASY).

Resting Blood Pressure: Resting blood pressure value (e.g., 120).

Cholesterol: Serum cholesterol in mg/dl (e.g., 200).

Fasting Blood Sugar: 1 if fasting blood sugar > 120 mg/dl, otherwise 0.

Resting ECG: Three options (Normal, ST, LVH).

Max Heart Rate: Maximum heart rate achieved during exercise (e.g., 150).

Exercise Angina: Y (Yes) or N (No).

Oldpeak: ST depression induced by exercise relative to rest (e.g., 0.5).

ST Slope: Slope of the peak exercise ST segment (Up, Flat, Down).

Date: The date the data is entered (e.g., 2024-12-10).

### Output

The system provides the following outputs:
1. Displays in the terminal (e.g., "Heart Disease Detected").
2. Saved Files：Health Records (e.g., Mia_data.csv).
3. Saved Files： Health Trends Visualization (e.g., Mia_health_trend.png).
4. Saved Files： Detailed Risk Report (e.g., Nora_diagnosis_history.csv).


### How to Use
1. Download the zip file, make sure heart.csv includes.
2. Run the last python file, Diagnosis.py.
3. Choose an option : 1. Input new data and get predictions; 2. View user health history; 3. Visualize health trends; 4. View diagnosis history.

Option 1: Input new health data and receive a prediction. Follow prompts to input data and save results.
   
Option 2: Input the name. View saved health records stored in the personal CSV file.

Option 3: Input the name. Generate and view health trend visualizations based on historical data.

Option 4: Input the name. View detailed diagnosis history, including abnormal values and risk levels.

If the information is for a new user, enter the data first (option1) before health history (option2), health trends (option3), and diagnosis history (option4). It is not recommended to enter data once to view health trends as there is no comparison.

If the user already has the relevant files, health history, health trends and diagnosis history, the functions can be called up directly.
If entering new data again, it is recommended to delete the previous health_trends.png and then use health trends(option3).

