import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pickle
import os

def train_and_save_model():
    """
    Loads the heart disease dataset, preprocesses it, trains a Logistic Regression model,
    and saves the trained model and preprocessor to 'model.pkl'.
    """
    # Define the path to the dataset
    data_path = os.path.join('data', 'heart.csv')

    # Check if the dataset exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please download 'heart.csv' and place it in the 'data' folder.")
        return

    # Load the dataset
    try:
        df = pd.read_csv(data_path)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # The target column is 'TenYearCHD' based on your provided columns
    target_column = 'TenYearCHD'
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the dataset.")
        print("Please check your 'heart.csv' file and update the script with the correct target column name.")
        return

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Define categorical and numerical features based on your provided columns
    categorical_features = [
        'male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke',
        'prevalentHyp', 'diabetes'
    ]
    numerical_features = [
        'age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
    ]

    # Ensure all defined features actually exist in the DataFrame
    missing_num_features = [f for f in numerical_features if f not in X.columns]
    missing_cat_features = [f for f in categorical_features if f not in X.columns]

    if missing_num_features or missing_cat_features:
        print(f"Error: Missing features in dataset.")
        if missing_num_features:
            print(f"Numerical features not found: {missing_num_features}")
        if missing_cat_features:
            print(f"Categorical features not found: {missing_cat_features}")
        print("Please ensure your 'heart.csv' file matches the expected column names.")
        return

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any, though ideally all are handled
    )

    # Create a pipeline that first preprocesses the data and then applies the model
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', LogisticRegression(solver='liblinear', random_state=42))])

    # Split data into training and testing sets
    # Drop rows with NaN values before splitting, as some datasets might have them
    df_cleaned = df.dropna(subset=numerical_features + categorical_features + [target_column])
    X_cleaned = df_cleaned.drop(target_column, axis=1)
    y_cleaned = df_cleaned[target_column]

    if X_cleaned.empty:
        print("Error: No valid data rows left after dropping NaNs. Please check your dataset for missing values.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42, stratify=y_cleaned)

    # Train the model
    print("Training model...")
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model (optional, for verification)
    accuracy = model_pipeline.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.2f}")

    # Save the trained model (the entire pipeline including preprocessor)
    model_filename = 'model.pkl'
    try:
        with open(model_filename, 'wb') as file:
            pickle.dump(model_pipeline, file)
        print(f"Model saved successfully as '{model_filename}'")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    train_and_save_model()
