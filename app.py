import os
from flask import Flask, render_template, request

from predict import IrisPredictor, FEATURE_COLS

app = Flask(__name__)

predictor = IrisPredictor(
    artifacts_dir=os.environ.get("ARTIFACTS_DIR", "artifacts")
)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probabilities = None
    error = None
    values = {}

    if request.method == "POST":
        try:
            values = {
                c: float(request.form[c]) for c in FEATURE_COLS
            }
            result = predictor.predict_one(values)
            prediction = result["prediction"]
            probabilities = result["probabilities"]
        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        features=FEATURE_COLS,
        prediction=prediction,
        probabilities=probabilities,
        error=error,
        values=values,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
