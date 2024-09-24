from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.logger import logging

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        # Extracting features from the form
        data = CustomData(
            length=request.form.get('Length'),
            diameter=request.form.get('Diameter'),
            height=request.form.get('Height'),
            whole_weight=request.form.get('Whole weight'),
            shucked_weight=request.form.get('Shucked weight'),
            viscera_weight=request.form.get('Viscera weight'),
            shell_weight=request.form.get('Shell weight'),
            rings=request.form.get('Rings')
        )
        
        # Convert input data to DataFrame
        final_new_data = data.get_data_as_dataframe()
        
        # Initialize prediction pipeline
        predict_pipeline = PredictPipeline()
        
        # Make prediction
        pred = predict_pipeline.predict(final_new_data)
        
        # Map prediction result to human-readable format
        if pred[0] == 0:
            result_label = "Male"
        elif pred[0] == 1:
            result_label = "Female"
        else:
            result_label = "Infant"
        
        logging.info(f'Prediction result: {result_label}')
        
        return render_template('results.html', final_result=result_label)

if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=True)
