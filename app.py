# app.py

from flask import Flask, render_template, request
import numpy as np
import pickle
from datetime import datetime

app = Flask(__name__)

# --- Load models & encoder ---

reg_model = pickle.load(open("delay_regression_model.pkl", "rb"))
cls_model = pickle.load(open("delay_classification_model.pkl", "rb"))
weather_encoder = pickle.load(open("weather_encoder.pkl", "rb"))

DELAY_THRESHOLD = 15

@app.route("/route-heatmap")
def heatmap():
    return render_template("heatmap.html")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        try:
            distance_km = float(request.form["distance_km"])
            temp_c = float(request.form["temp_c"])
            weather_text = request.form["weather"]
            travel_date = request.form["date"]  # YYYY-MM-DD

            # Date → month & day_of_week
            dt = datetime.strptime(travel_date, "%Y-%m-%d")
            month = dt.month
            day_of_week = dt.weekday()

            # Weather encode
            weather_encoded = weather_encoder.transform([weather_text])[0]

            # Features: [distance_km, temp_c, weather_encoded, month, day_of_week]
            features = np.array([[distance_km, temp_c, weather_encoded, month, day_of_week]])

            # ML predictions
            predicted_delay = reg_model.predict(features)[0]
            prob_delay = cls_model.predict_proba(features)[0][1] * 100
            delayed_label = cls_model.predict(features)[0]

            result = {
                "predicted_delay": round(predicted_delay, 2),
                "prob_delay": round(prob_delay, 2),
                "delayed_label": int(delayed_label),
                "threshold": DELAY_THRESHOLD,
            }

        except Exception as e:
            result = {"error": str(e)}

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
