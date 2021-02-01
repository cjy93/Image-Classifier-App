import logging
from ..src.inference import Inference
import numpy as np
#import pytest
import os
from PIL import Image

model_path =  os.path.join(os.path.dirname(__file__),"../tensorfood.h5")
image_path =  os.path.join(os.path.dirname(__file__),"../images/curry_puff.jpg")

model = Inference()
'''
do no put any arguments in your functions
'''
def test_load_model():
    assert model_path.endswith(('.h5')), "Please upload a '.h5' file"
    
def test_check_predict():
    prediction, prob = model.predict_food(image_path)
    assert prediction == 'wanton_noodle', "wrong prediction!"
    
def test_check_image_mode():
    mode_change = ['CMYK','RGBA', 'P','RGB'] 
    img = Image.open(image_path)
    assert img.mode in mode_change,"please provide file modes of ['CMYK','RGBA', 'P','RGB'], so we can atleast change to 'RGB'"
        
# Meeting: continuous delivery, continuous integration: Push branch to master, not to your own branch
# code in teams cannot break other people code, and also need documentation
# (yourenvname) C:\Users\jia yi\Desktop\aiap_new\all-assignments\assignment7>pytest