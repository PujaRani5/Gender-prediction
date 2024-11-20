import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Prepare the data
X = data[['Height', 'Weight']]  # Features: height and weight
y = data['Sex']  # Target: gender

# Convert gender labels to binary (0 for Female, 1 for Male)
y = y.map({'F': 0, 'M': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# Function to predict gender for new data
def predict_gender(height, weight):
    prediction = model.predict([[height, weight]])
    return 'Male' if prediction[0] == 1 else 'Female'

# Example usage
height = input("Enter the Height in cm ")  # Replace with your input
weight = input("Enter the Weight in kg ")   # Replace with your input
print(f"The predicted gender for height {height} cm and weight {weight} kg is: {predict_gender(height, weight)}")
