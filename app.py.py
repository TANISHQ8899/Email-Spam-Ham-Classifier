from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

MODEL_DIR = "C:/Users/tanis/OneDrive/Desktop/Machine Learnng/mail detection"

def load_pickle(file_name):
    file_path = os.path.join(MODEL_DIR, file_name)
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

# Load the saved model and feature extraction
model = load_pickle('logistic_regression.pkl')
feature_extraction = load_pickle('feature_extraction.pkl')

def predict_mail(input_text):
    input_user_mail = [input_text]
    input_data_features = feature_extraction.transform(input_user_mail)
    prediction = model.predict(input_data_features)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def analyze_mail():
    classify = None
    if request.method == 'POST':
        mail = request.form.get('mail')
        if mail:
            classify = predict_mail(input_text=mail)
    return render_template('index.html', classify=classify)

if __name__ == '__main__':
    app.run(debug=True)
