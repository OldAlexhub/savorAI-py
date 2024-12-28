from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import gc
from functools import lru_cache
from dotenv import load_dotenv
import os
import pymongo

# Load environment variables
load_dotenv('.env')

app = Flask(__name__)
CORS(app)

# Lazy Load MongoDB Connection
def get_db():
    mongo_url = os.getenv('MONGO_URL')
    client = pymongo.MongoClient(mongo_url)
    return client['test']

# Lazy Load Datasets
def get_processed_data():
    db = get_db()
    return pd.DataFrame(list(db['processed_data'].find())).drop('_id', axis=1)

def get_data():
    db = get_db()
    return pd.DataFrame(list(db['data'].find())).drop('_id', axis=1)

@lru_cache(maxsize=1)
def get_transformer_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@lru_cache(maxsize=1)
def get_rf_model():
    processed_data = get_processed_data()
    X = processed_data.drop('name', axis=1)
    y = processed_data['name']
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

@app.route('/', methods=['GET'])
def get_home():
    return 'Hello Hungry World!'

@app.route('/predict', methods=['POST'])
def get_prediction():
    try:
        user_input = request.json

        # Parse user input
        input_data = pd.DataFrame([{
            'cuisine': user_input.get('cuisine', ''),
            'course': user_input.get('course', ''),
            'diet': user_input.get('diet', ''),
            'prep_time': float(user_input.get('prep_time', 0)),
            'cook_time': float(user_input.get('cook_time', 0))
        }])

        # Load Models
        transformer_model = get_transformer_model()
        rf_model = get_rf_model()

        # Encode String Columns
        for col in ['cuisine', 'course', 'diet']:
            if input_data[col].iloc[0]:
                input_data[col] = transformer_model.encode([input_data[col].iloc[0]])[0]

        # Match Input Columns with Training Data
        processed_data = get_processed_data()
        X = processed_data.drop('name', axis=1)
        input_data = input_data[X.columns]

        # Make Prediction
        prediction = rf_model.predict(input_data)[0]

        # Fetch the matching record directly from MongoDB
        db = get_db()
        result = db['data'].find_one({"name": prediction})

        if not result:
            return jsonify({'error': 'No matching result found'}), 404

        # Remove MongoDB ObjectId if present
        result.pop('_id', None)

        return jsonify(result)

    except Exception as e:
        gc.collect()
        return jsonify({'error': 'An error occurred', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
