import joblib

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def make_prediction(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def target_names():
    return ['setosa', 'versicolor', 'virginica']

def main(model_path='model.joblib', input_data=[[5.1, 3.5, 1.4, 0.2]]):
    model = load_model(model_path)
    prediction = make_prediction(model, input_data)
    predicted_class = target_names()[prediction[0]]
    print(f"Prediction: {predicted_class}")

if __name__ == "__main__":
    main(model_path='E:/my_projects/build_it/model/random_forest_model.joblib')