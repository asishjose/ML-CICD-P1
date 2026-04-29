import pickle

from flask import Flask, jsonify, request
from sklearn.datasets import load_iris


app = Flask(__name__)

with open("model/iris_model.pkl", "rb") as f:
    model = pickle.load(f)

iris = load_iris()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]

    prediction = model.predict([features])[0]

    return jsonify(
        {
            "prediction": int(prediction),
            "class_name": iris.target_names[prediction],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
