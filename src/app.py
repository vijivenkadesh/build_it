from flask import Flask, request, jsonify, render_template
from prediction import load_model, make_prediction, target_names
import logging
import os
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# template_dir = os.path.join(BASE_DIR, '..', 'templates')
# static_dir = os.path.join(BASE_DIR, '..', 'static')
# model_path = os.path.join(BASE_DIR, '..', 'model', 'random_forest_model.joblib')

# app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
# app = Flask(__name__, template_folder='E:/my_projects/build_it/templates', static_folder='E:/my_projects/build_it/static')
app = Flask(__name__, template_folder='templates', static_folder='static')
logging.info("Loading model...")
# model = load_model('E:/my_projects/build_it/model/random_forest_model.joblib')
model = load_model('model/random_forest_model.joblib')
# model = load_model(model_path)


@app.route('/')
def home():
    name = "oddsoul"
    return render_template('index.html', name=name)

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    elif request.method == 'POST':
        input_data = [request.form.getlist('sepal_length') + request.form.getlist('sepal_width') + request.form.getlist('petal_length') + request.form.getlist('petal_width')]
    # input_data = request.json.get('input_data')
    logging.info(f"Received input data: {input_data}")
    prediction = make_prediction(model, input_data)
    predicted_class = target_names()[prediction[0]]
    logging.info(f"Prediction made: {predicted_class}")
    # return jsonify({'prediction': predicted_class})
    return render_template('predict.html', prediction=predicted_class)
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)