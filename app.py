from flask import Flask,render_template, request
import os
import numpy as np
import pandas as pd
from src.datascience.pipeline.predictions_pipeline import PredictionPipeline


app=Flask(__name__)

@app.route('/',methods=['GET']) # route to display the home page
def homepage():
    return render_template('index.html')



@app.route('/train',methods=['GET']) #route to train the pipeline
def trainig():
    os.system('python main.py')
    return "Training Successful"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Recoge los datos en formato JSON
            data = request.get_json()

            # Convierte los datos a valores flotantes
            fixed_acidity = float(data['fixed_acidity'])
            volatile_acidity = float(data['volatile_acidity'])
            citric_acid = float(data['citric_acid'])
            residual_sugar = float(data['residual_sugar'])
            chlorides = float(data['chlorides'])
            free_sulfur_dioxide = float(data['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(data['total_sulfur_dioxide'])
            density = float(data['density'])
            pH = float(data['pH'])
            sulphates = float(data['sulphates'])
            alcohol = float(data['alcohol'])

            # Prepara los datos para la predicción
            input_data = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                   free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]).reshape(1, -1)

            # Realiza la predicción
            obj = PredictionPipeline()
            prediction = obj.predict(input_data)

            rounded_prediction = round(float(prediction[0]),1)

            # Devuelve la predicción como respuesta JSON
            return {"prediction": rounded_prediction}, 200

        except Exception as e:
            # Devuelve un mensaje de error en caso de fallo
            return {"error": "Something went wrong", "details": str(e)}, 500

    

if __name__ =='__main__':
    app.run(host='0.0.0.0',port=8080)