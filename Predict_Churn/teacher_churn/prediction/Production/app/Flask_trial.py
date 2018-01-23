from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators

app = Flask(__name__)

@app.route('/')
@app.route('/index')


def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def predict():
    return render_template('index.html')
