from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the model and preprocessor
model_path = 'cookie_classifier.joblib'
model_data = joblib.load(model_path)
trained_model = model_data['model']
preprocessor = model_data['preprocessor']

def preprocess_df(preprocessor, df):  
    input_data_transformed = preprocessor.transform(df)
    return input_data_transformed

def make_predictions(model, input_data_transformed):
    # Make predictions
    predictions = model.predict(input_data_transformed)
    predicted_category_index = np.argmax(predictions, axis=1)

    try:
        # Map the category index to the category name
        category_mapping = {0: "Functionality", 1 : 'Performance', 2: 'Strictly Necessary', 3: 'Targeting/Advertising'}
        predicted_categories = [category_mapping.get(idx) for idx in predicted_category_index]
    except:
        return "Unknown"

    return predicted_categories

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()
        df = pd.DataFrame(data)

        # Preprocess the data
        input_data_transformed = preprocess_df(preprocessor, df)

        # Convert NumPy array to a list
        input_data_transformed_list = input_data_transformed.tolist()

        # Make predictions
        predictions = make_predictions(trained_model, input_data_transformed_list)

        # Return the predictions as JSON
        result = {"predictions": predictions}
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
