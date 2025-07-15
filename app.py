from flask import Flask, request, render_template
import joblib
import numpy as np

# Load trained model
model = joblib.load("math_score_predictor.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        try:
            gender = int(request.form["gender"])
            race = int(request.form["race"])
            education = int(request.form["education"])
            lunch = int(request.form["lunch"])
            prep = int(request.form["prep"])
            reading = float(request.form["reading"])
            writing = float(request.form["writing"])

            # Prepare input
            features = np.array([[gender, race, education, lunch, prep, reading, writing]])
            prediction = round(model.predict(features)[0], 2)
        except Exception as e:
            prediction = f"Error: {e}"
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
