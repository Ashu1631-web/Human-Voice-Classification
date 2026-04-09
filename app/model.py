import joblib

model = joblib.load("models/classifier.pkl")

def predict_voice(features):
    return model.predict(features)[0]
