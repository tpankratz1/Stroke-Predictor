import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template
from sklearn import preprocessing

app=Flask(__name__)
model=pickle.load(open('model.pkl', 'rb'))

cols=['Percent_Smokers','Average_Daily_Air_Polution_Particle_Matter','Food_Environment_Index','Percent_Unemployed','Income_Inequality_Ratio']

@app.route('/')
def home():
    feature_dict=request.form.to_dict()
    return render_template('index.html', feature_dict=feature_dict)

@app.route('/predict',methods=['POST'])
def predict():

    feature_dict=request.form.to_dict()

    f_items=list(feature_dict.values())

    f_items=list(map(float, f_items))

    final_items=np.array(f_items).reshape(1, 5)

    prediction=model.predict(final_items)
    
    output=round(prediction[0], 1)

    return render_template('index.html', prediction_text='{} deaths'.format(output), feature_dict=feature_dict)

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='localhost', port=5000)

# Reference/Source: https://towardsdatascience.com/machine-learning-model-deployment-on-heroku-using-flask-467acb4a34da