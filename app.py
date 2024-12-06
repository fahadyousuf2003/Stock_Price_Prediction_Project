from flask import Flask, render_template, request, redirect, url_for, jsonify
from smPredictor.pipeline.prediction import PredictionPipeline
from smPredictor.entity.config_entity import EvaluationConfig
import subprocess
import os
import threading
import time

app = Flask(__name__)


def train_models():
    """Background thread to run pipeline"""
    global training_in_progress
    try:
        training_in_progress = True
        
        # Run main.py with live output
        process = subprocess.Popen(['python', 'main.py'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.STDOUT, 
                                   universal_newlines=True,
                                   cwd=os.getcwd())
        
        # Stream output to terminal in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Check return code
        if process.returncode != 0:
            raise Exception(f"Pipeline failed with return code {process.returncode}")
    
    except Exception as e:
        print(f"Pipeline error: {e}")
    finally:
        training_in_progress = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train-model', methods=['POST'])
def train_model():
    """Route to start model training"""
    global training_in_progress
    try:
        # Start training in a background thread
        training_thread = threading.Thread(target=train_models)
        training_thread.start()
        
        # Render training in progress page
        return render_template('training.html')
    
    except Exception as e:
        print(f"Training initiation error: {str(e)}")
        return render_template('error.html', error=str(e)), 500

@app.route('/check-training-status')
def check_training_status():
    """Route to check training status"""
    return jsonify({
        'status': 'in_progress' if training_in_progress else 'completed'
    })
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        category = request.form['category']
        stock = request.form['stock']
        days = int(request.form['days'])

        # Validate inputs
        if days < 1 or days > 365:
            raise ValueError("Prediction days must be between 1 and 365")

        # Create a basic EvaluationConfig
        config = EvaluationConfig(
            apple_raw_data_dir='artifacts/data_ingestion/international_stocks/raw_data/AAPL.csv',
            amazon_raw_data_dir='artifacts/data_ingestion/international_stocks/raw_data/AMZN.csv',
            google_raw_data_dir='artifacts/data_ingestion/international_stocks/raw_data/GOOG.csv',
            microsoft_raw_data_dir='artifacts/data_ingestion/international_stocks/raw_data/MSFT.csv',
            silk_raw_data_dir='artifacts/data_ingestion/pakistan_stocks/raw_data/SILK.csv',
            pace_raw_data_dir='artifacts/data_ingestion/pakistan_stocks/raw_data/PACE.csv',
            fauji_raw_data_dir='artifacts/data_ingestion/pakistan_stocks/raw_data/FFL.csv',
            punjab_raw_data_dir='artifacts/data_ingestion/pakistan_stocks/raw_data/BOP.csv',
            apple_model_dir="artifacts/model_trainer/apple_model.keras",
            amazon_model_dir="artifacts/model_trainer/amazon_model.keras",
            google_model_dir="artifacts/model_trainer/google_model.keras",
            microsoft_model_dir="artifacts/model_trainer/microsoft_model.keras",
            silk_model_dir="artifacts/model_trainer/silk_model.keras",
            pace_model_dir="artifacts/model_trainer/pace_model.keras",
            fauji_model_dir="artifacts/model_trainer/fauji_model.keras",
            punjab_model_dir="artifacts/model_trainer/punjab_model.keras"
        )

        # Create prediction pipeline with the config
        pipeline = PredictionPipeline(config)

        # Get predictions
        predictions = pipeline.predict(days, category, stock)
        
        # Render a simple template to display predictions
        return render_template('predictions.html', 
                               predictions=predictions.to_dict('records'), 
                               category=category.capitalize(), 
                               stock=stock.capitalize())
    
    except Exception as e:
        # Log the error 
        print(f"Prediction error: {str(e)}")
        
        # Render error page
        return render_template('error.html', error=str(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)