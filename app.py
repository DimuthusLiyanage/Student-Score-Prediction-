from flask import Flask, request, render_template
import xgboost as xgb
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1. Load the trained model
model = xgb.Booster()
model.load_model("exam_score_xgb_model.json")

# 2. Get the feature names the model was trained on
# This is crucial to ensure our input matches the training data exactly
MODEL_FEATURES = model.feature_names

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 3. Get data from the HTML form
        data = {
            'age': float(request.form['age']),
            'study_hours': float(request.form['study_hours']),
            'class_attendance': float(request.form['class_attendance']),
            'sleep_hours': float(request.form['sleep_hours']),
            
            # Categorical variables
            'gender': request.form['gender'],
            'course': request.form['course'],
            'internet_access': request.form['internet_access'],
            'sleep_quality': request.form['sleep_quality'],
            'study_method': request.form['study_method'],
            'facility_rating': request.form['facility_rating'],
            'exam_difficulty': request.form['exam_difficulty']
        }

        # 4. Convert to DataFrame
        df = pd.DataFrame([data])

        # 5. Apply One-Hot Encoding (Same as your notebook)
        # We perform get_dummies to convert text to columns
        df_processed = pd.get_dummies(df)

        # 6. ALIGNMENT (The most important step!)
        # We ensure the dataframe has exactly the same columns as the trained model.
        # If a column is missing (e.g., 'gender_male'), it gets added with 0.
        # If extra columns exist, they are removed.
        df_final = df_processed.reindex(columns=MODEL_FEATURES, fill_value=0)

        # 7. Convert to DMatrix for XGBoost
        dmatrix_data = xgb.DMatrix(df_final)

        # 8. Predict
        prediction = model.predict(dmatrix_data)[0]
        
        # Format the result (limit 0-100)
        final_score = max(0, min(100, round(prediction, 2)))

        return render_template('index.html', prediction_text=f'Predicted Exam Score: {final_score}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)