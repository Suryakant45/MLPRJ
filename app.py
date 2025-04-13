import sys
import os
import traceback
from src.DiamondPricePrediction.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.DiamondPricePrediction.logger import logging
from flask import Flask, request, render_template, jsonify

# Initialize Flask app
app = Flask(__name__)

# Configure logging
@app.before_first_request
def setup_logging():
    if not app.debug:
        # Set up your logging configuration here
        pass

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    try:
        if request.method == "GET":
            logging.info("Rendering form.html for GET request")
            return render_template("form.html")

        else:
            logging.info("Processing POST request with form data")
            # Validate form data
            for field in ['carat', 'depth', 'table', 'x', 'y', 'z']:
                if not request.form.get(field):
                    error_msg = f"Missing required field: {field}"
                    logging.error(error_msg)
                    return render_template("error.html", error=error_msg), 400

            # Create CustomData object
            try:
                data = CustomData(
                    carat=float(request.form.get('carat')),
                    depth=float(request.form.get('depth')),
                    table=float(request.form.get('table')),
                    x=float(request.form.get('x')),
                    y=float(request.form.get('y')),
                    z=float(request.form.get('z')),
                    cut=request.form.get('cut'),
                    color=request.form.get('color'),
                    clarity=request.form.get('clarity')
                )
                logging.info("CustomData object created successfully")
            except ValueError as ve:
                error_msg = f"Invalid input data: {str(ve)}"
                logging.error(error_msg)
                return render_template("error.html", error=error_msg), 400

            # Convert to dataframe
            final_data = data.get_data_as_dataframe()
            logging.info("Dataframe created successfully")

            # Make prediction
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_data)
            logging.info(f"Prediction successful: {pred}")

            result = round(pred[0], 2)
            logging.info(f"Final result: {result}")

            return render_template("result.html", final_result=result)

    except Exception as e:
        error_msg = f"Error in predict_datapoint: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        return render_template("error.html", error=error_msg), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Internal server error'), 500

# Main execution
if __name__ == '__main__':
    try:
        # Check if required directories and files exist
        if not os.path.exists('templates'):
            raise FileNotFoundError("Templates directory not found")

        if not os.path.exists('artifacts'):
            raise FileNotFoundError("Artifacts directory not found")

        required_templates = ['index.html', 'form.html', 'result.html', 'error.html']
        for template in required_templates:
            if not os.path.exists(os.path.join('templates', template)):
                raise FileNotFoundError(f"Template file not found: {template}")

        required_artifacts = ['model.pkl', 'preprocessor.pkl']
        for artifact in required_artifacts:
            if not os.path.exists(os.path.join('artifacts', artifact)):
                raise FileNotFoundError(f"Artifact file not found: {artifact}")

        # Start the Flask app
        print("Starting Flask application on port 80...")
        app.run(host="0.0.0.0", port=80, debug=False)

    except Exception as e:
        print(f"Error starting application: {str(e)}")
        traceback.print_exc()

