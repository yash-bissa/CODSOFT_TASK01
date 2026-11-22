"""CodSoft SMS Spam Detection.

Install: python -m pip install flask pandas scikit-learn
Place spam.csv next to this file. It must have v1/v2 or label/message columns.
Run: python codsoft_task1_sms_spam.py
Open: http://127.0.0.1:5000
"""

from pathlib import Path

import pandas as pd
from flask import Flask, render_template_string, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

DATASET = Path(__file__).with_name("spam.csv")
app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SMS Spam Detector</title>
<style>
* { box-sizing: border-box; }
body { margin: 0; padding: 3rem 1rem; background: #f1f5f9;
       color: #172038; font: 1rem/1.5 system-ui, sans-serif; }
main { max-width: 38rem; margin: auto; background: white; padding: 2rem;
       border: 1px solid #dbe4ef; border-radius: 1rem; }
h1 { margin: 0; }
label { display: block; font-weight: 600; }
textarea { display: block; width: 100%; min-height: 8rem; margin: .5rem 0 1rem;
           padding: .75rem; border: 1px solid #64748b; border-radius: .5rem;
           font: inherit; resize: vertical; }
button { padding: .7rem 1rem; border: 0; border-radius: .5rem;
         color: white; background: #1d4ed8; font: inherit; cursor: pointer; }
.result { padding: .8rem; background: #eaf2ff; border-radius: .5rem; }
</style>
</head>
<body>
<main>
  <h1>SMS Spam Detector</h1>
  <p>Enter a message to classify it as spam or ham (legitimate).</p>
  <form method="post">
    <label for="message">SMS message</label>
    <textarea id="message" name="message" required maxlength="5000">{{ message }}</textarea>
    <button type="submit">Classify message</button>
  </form>
  {% if result %}<p class="result" role="status">{{ result }}</p>{% endif %}
</main>
</body>
</html>
"""


def train_model():
    try:
        data = pd.read_csv(DATASET, encoding="utf-8-sig")
    except UnicodeDecodeError:
        data = pd.read_csv(DATASET, encoding="latin-1")

    data.columns = data.columns.str.strip().str.lower()
    data = data.rename(columns={"v1": "label", "v2": "message"})
    if not {"label", "message"}.issubset(data.columns):
        raise ValueError("spam.csv needs v1/v2 or label/message columns.")

    data = data[["label", "message"]].dropna().copy()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].astype(str).str.strip()
    data = data[data["label"].isin(["ham", "spam"]) & data["message"].ne("")]
    data = data.drop_duplicates(subset="message")

    counts = data["label"].value_counts()
    if counts.get("ham", 0) < 5 or counts.get("spam", 0) < 5:
        raise ValueError("Provide at least five unique messages per class.")

    x_train, x_test, y_train, y_test = train_test_split(
        data["message"], data["label"],
        test_size=0.2, random_state=42, stratify=data["label"],
    )
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"),
        MultinomialNB(),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(classification_report(y_test, predictions, zero_division=0))
    return model


@app.route("/", methods=["GET", "POST"])
def home():
    message = request.form.get("message", "").strip()
    result = ""
    if request.method == "POST":
        if not message or len(message) > 5000:
            return render_template_string(
                PAGE, message=message[:5000],
                result="Enter a message between 1 and 5,000 characters.",
            ), 400
        label = app.config["SMS_MODEL"].predict([message])[0]
        result = "Prediction: Spam" if label == "spam" else "Prediction: Ham (legitimate)"
    return render_template_string(PAGE, message=message, result=result)


if __name__ == "__main__":
    try:
        app.config["SMS_MODEL"] = train_model()
    except FileNotFoundError:
        raise SystemExit("Place spam.csv next to this Python file.") from None
    except ValueError as error:
        raise SystemExit(str(error)) from None
    app.run()
