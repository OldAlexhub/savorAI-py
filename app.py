import os
import pymongo
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import gc

# Load environment variables
load_dotenv('.env')

# MongoDB Connection
mongo_url = os.getenv('MONGO_URL')
client = pymongo.MongoClient(mongo_url)
db = client['test']


# Lazy Loading Datasets
def get_processed_data():
    return pd.DataFrame(list(db['processed_data'].find())).drop('_id', axis=1)


def get_data():
    return pd.DataFrame(list(db['data'].find())).drop('_id', axis=1)


# Lazy Load Models with Caching
@lru_cache(maxsize=1)
def get_transformer_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Lightweight model


@lru_cache(maxsize=1)
def get_rf_model():
    processed_collection = get_processed_data()
    X = processed_collection.drop('name', axis=1)
    y = processed_collection['name']
    model = RandomForestClassifier(n_estimators=25, random_state=42)
    model.fit(X, y)
    return model


# Flask App Initialization
app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def get_home():
    return 'Hello Hungry World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    try:
        user_input = request.json

        # Validate and Parse User Input
        try:
            input_data = pd.DataFrame([{
                'cuisine': user_input.get('cuisine', ''),
                'course': user_input.get('course', ''),
                'diet': user_input.get('diet', ''),
                'prep_time': float(user_input.get('prep_time', 0)),
                'cook_time': float(user_input.get('cook_time', 0))
            }])
        except (ValueError, TypeError) as e:
            return jsonify({'error': 'Invalid input data', 'details': str(e)}), 400

        # Load Models
        transformer_model = get_transformer_model()
        rf_model = get_rf_model()

        # Encode String Columns
        for col in ['cuisine', 'course', 'diet']:
            if input_data[col].iloc[0]:
                input_data[col] = transformer_model.encode([input_data[col].iloc[0]])[0]

        # Ensure Column Order Matches Training Data
        processed_collection = get_processed_data()
        X = processed_collection.drop('name', axis=1)
        input_data = input_data[X.columns]

        # Make Prediction
        prediction = rf_model.predict(input_data)

        # Fetch Results
        data_collection = get_data()
        result = data_collection[data_collection['name'] == prediction[0]]
        if result.empty:
            return jsonify({'error': 'No matching result found'}), 404

        result_dict = result.iloc[0].to_dict()

        # Clear unused memory
        gc.collect()

        return jsonify(result_dict)

    except Exception as e:
        gc.collect()
        return jsonify({'error': 'An error occurred', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
