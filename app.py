import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
cols = ['Percent_Smokers','Average_Daily_Air_Polution_Particle_Matter','Food_Environment_Index','Percent_Unemployed','Income_Inequality_Ratio']

@app.route('/')
def home():
    feature_dict = request.form.to_dict()
    return render_template('index.html', feature_dict=feature_dict)

@app.route('/predict',methods=['POST'])
def predict():

    feature_dict = request.form.to_dict()
    feature_list = list(feature_dict.values())
    feature_list = list(map(float, feature_list))
    final_features = np.array(feature_list).reshape(1, 5)

    prediction = model.predict(final_features)
    output = round(prediction[0], 1)

    return render_template('index.html', prediction_text='{} deaths'.format(output), feature_dict=feature_dict)

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='localhost', port=5000)

# Reference/Source: https://towardsdatascience.com/machine-learning-model-deployment-on-heroku-using-flask-467acb4a34da