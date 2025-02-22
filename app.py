import numpy as np
from flask import Flask, render_template, jsonify, request

import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #convert values to int format
    int_features=[int(x) for x in request.form.values()]
    #fill that values to array format
    final_features=[np.array(int_features)]
    #now predicting
    prediction=model.predict(final_features)
    #for output
    output=round(prediction[0],2)
    return render_template('index.html',prediction_text="Employee salary should be $ {}".format(output))


@app.route('/prediction_api',methods=['POST'])
def predict_api():
     '''
    For direct API calls trought request
    '''
     data=request.get_json(force=True)
     prediction=model.predict([np.array(list(data.values()))])
     output=prediction[0]
     return jsonify(output)
if __name__=="__main__":
    app.run(debug=True)

