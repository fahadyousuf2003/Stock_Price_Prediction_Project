from flask import Flask, render_template, request
from smPredictor.pipeline.prediction import PredictionPipeline
from smPredictor.entity.config_entity import EvaluationConfig

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

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
    app.run(debug=True)