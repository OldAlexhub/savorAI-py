import os
import pymongo
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv('.env')

# MongoDB Connection
mongo_url = os.getenv('MONGO_URL')
client = pymongo.MongoClient(mongo_url)
db = client['test']

# Load Data from MongoDB
processed_collection = pd.DataFrame(list(db['processed_data'].find())).drop('_id', axis=1)
data_collection = pd.DataFrame(list(db['data'].find())).drop('_id', axis=1)

# Model Initialization
model = SentenceTransformer('all-MiniLM-L6-v2')
rfmodel = RandomForestClassifier(n_estimators=5, random_state=42)

# Data Preparation
X = processed_collection.drop('name', axis=1)
y = processed_collection['name']

# Train the Random Forest Model
rfmodel.fit(X, y)

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

        # Encode String Columns
        if input_data['cuisine'].iloc[0]:
            input_data['cuisine'] = model.encode([input_data['cuisine'].iloc[0]])
        if input_data['course'].iloc[0]:
            input_data['course'] = model.encode([input_data['course'].iloc[0]])
        if input_data['diet'].iloc[0]:
            input_data['diet'] = model.encode([input_data['diet'].iloc[0]])

        # Ensure Column Order Matches Training Data
        input_data = input_data[X.columns]

        # Make Prediction
        prediction = rfmodel.predict(input_data)

        # Fetch Results
        result = data_collection[data_collection['name'] == prediction[0]]
        if result.empty:
            return jsonify({'error': 'No matching result found'}), 404

        result_dict = result.iloc[0].to_dict()

        return jsonify(result_dict)

    except Exception as e:
        return jsonify({'error': 'An error occurred', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)