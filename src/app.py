from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import mistune
import PIL
from PIL import Image
import logging



# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify, flash
from werkzeug.utils import secure_filename

# import the model
from .inference import Inference

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
inference = Inference()

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

print('Model loaded. Check http://127.0.0.1:8000/')


with open("README.md", "r") as f:
    readme_doc = f.read()




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
    
@app.route('/docs', methods=['GET'])
def docs():
    # Main page
    return render_template("docs.html", data=mistune.markdown(readme_doc))



@app.route('/info', methods=['GET'])
def short_description():
    # Serialize the result, you can add additional fields
    return jsonify(**inference.info())


@app.route('/predict', methods=['GET', 'POST'])
# means can accept GET and POST method
def predict():
    app.logger.info('Image file received')
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']  # https://stackoverflow.com/questions/54418594/python-flask-and-werkzeug-keep-giving-badrequestkeyerror-400-bad-request
                                    # https://stackoverflow.com/questions/3111779/how-can-i-get-the-file-name-from-request-files
         
        img = Image.open(f)
        '''
        Gitlab cannot recognise the path to your pictures. So change 
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, '../images', secure_filename(f.filename))
        f.save(file_path)
        '''
        # Make prediction
        
        try:
            # pass the image path to the prediction function
            predicted_food, predict_prob = inference.predict(img) #(file_path)
            predict_prob = str(predict_prob)
            predicted_food = predicted_food.replace("_", " ").capitalize()
            result = f'{predicted_food} with Probability:{predict_prob}'
        #except PIL.UnidentifiedImageError:
            #result = "Invalid Image"
            #proba = ""
        except FileNotFoundError:
            result = "File not found"
            proba = ""
        except ValueError: 
            result = "Invalid Image type"
            proba = ""
        except FileNotFoundError:
            result = "File Not Found"
            proba = ""
        except Exception as e:
            logging.error(traceback.format_exc())
            result = "Please input valid file path"
            proba = ""
        except:
            result = "Wrong image mode type"
            proba = ""
        #app.logger.info(f"Predicted:{predicted_food}, Probability:{predict_prob}")
        return result
    return None


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8000)
    # For production mode, comment the line above and uncomment below
    # serve(app, host="0.0.0.0", port=8000)
