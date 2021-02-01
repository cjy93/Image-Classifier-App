import logging
import argparse
#from polyaxon_client.tracking import Experiment, get_log_level, get_outputs_path
from PIL import Image
import os
import numpy as np
import tensorflow as tf

#logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)
#logger = logging.getLogger("polyaxon")
#experiment = Experiment()

         
class Inference:
    def __init__(self):
        # image_path = "../tensorflow.h5"
        # init your model here
        '''
        '''
        #self.image_path = os.path.join(os.path.dirname(__file__), image_path)
        
        self.foods = ['chilli_crab',
         'curry_puff',
         'dim_sum',
         'ice_kacang',
         'kaya_toast',
         'nasi_ayam',
         'popiah',
         'roti_prata',
         'sambal_stingray',
         'satay',
         'tau_huay',
         'wanton_noodle']
        self.model_path = os.path.join(os.path.dirname(__file__), '../tensorfood.h5')
        
         
        #self.best_class_index = None
        #self.image_path = image_path
   
         
    def _check_image_type(self, image_path):
        '''
        Check whether it is {jpeg, jpg, png}. If it is not, throw an error to tell then to upload correct format
        
        args: image_path
        returns: image PIL file im
        '''
        assert image_path.endswith(('.png', '.jpg', '.jpeg')), "Please upload file of type '.png','jpeg','jpg'"
        im = Image.open(image_path) # -> PIL image
        return im
        
    def _check_image_mode(self, img):
        '''
        check whether it is 'RGB'. If it is not RGB, please change to RGB and save as a .png file
        If it is not even in ['CMYK','RGBA', 'P'], then we throw assertion error as other file types cannot convert to RGB
        
        args: PIL file img
        returns: img      
        
        '''  
        mode_change = ['CMYK','RGBA', 'P']                
        #convert other modes to RGB
        if (img.mode in mode_change):
            img = img.convert('RGB')
            img.save(image_path) # over write file as RGB
        return img
        
        
    def _resize_image(self,im):
        '''
        first get the image shape based on the pretrained model. Then reshape 
        '''
        model = tf.keras.applications.MobileNet(input_shape=(160,160,3),include_top=True, weights='imagenet', classes=1000)
        # check the input format
        dims = model.input_shape[1:3] # -> (height, width)
        # use the dim as target shape for resize_image
        #im = tf.keras.preprocessing.image.load_img(image_path, target_size=dims) # -> PIL image
        im = im.resize(dims, Image.ANTIALIAS) # https://github.com/python-pillow/Pillow/issues/3360
        # convert image to tensor array
        doc = tf.keras.preprocessing.image.img_to_array(im) # -> numpy array
        doc = np.expand_dims(doc, axis=0)
        return doc
              
        
                
    def _check_filemode(self):
        '''
        only accepts file ending with .h5
        '''
        assert self.model_path.endswith(('.h5')), "Please upload file of type '.h5'"
        
        
        
    def _load_model(self):
        model = tf.keras.models.load_model(self.model_path)
        return model 
        
        
    def info(self):
        dict = {}
        model = self._load_model()
        dict['model_shape'] = model.input_shape[1:4]
        dict['model_name'] = 'MobileNetV2'
        dict['weights'] = 'imagenet'
        dict['num_of_classes'] = 12
        return dict
        
    def predict(self, im):
        ''' 
        this is a helper function for predict function later. This is required in app.py since the input arg should be img file
        args: img pillow image
        return: predicted category
        '''
        img = self._check_image_mode(im)
        doc = self._resize_image(img)
        self._check_filemode()
        model = self._load_model()
        
        # predict the class index corresponding to highest softmax
        predicted_class = model.predict_classes(doc)
        print(model.predict_classes(doc))
        predicted_food = self.foods[predicted_class[0]] # to get the value out of the list 
        print("predicted_food:", predicted_food)
        predict_prob = model.predict(doc)[0][predicted_class[0]]
        print(model.predict(doc))        
        print("predicted prob:", predict_prob)
        return predicted_food, predict_prob
        
    def predict_food(self, image_path):
        img = self._check_image_type(image_path) # not using self.image_path
        predicted_food, predict_prob = self.predict(img)   
        return predicted_food, predict_prob
        
    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # add all possible parsers first here
    parser.add_argument('--image_path', type = str)
    # concat all the flags we passed in as parser object
    args = parser.parse_args()
    print(f'currentpath:{os.path.dirname(__file__)}')
    #image_path = os.path.join(os.path.dirname(__file__),'images/curry_puff.jpg')
    #print(f'image_path{image_path}')
    print(args.image_path)
    inference = Inference()
    predicted_class = inference.predict_food(args.image_path)
    
       
    # comment these 4 lines out if you need to test locally
    #experiment.log_metrics(Predicted_class=predicted_class)
    #experiment.log_metrics(accuracy=accuracy)
  
  
  # (yourenvname) C:\Users\jia yi\Desktop\aiap_new\all-assignments\assignment7>python -m src.inference --image_path "C:\Users\jia yi\Desktop\aiap_new\all-assignments\assignment7\images/curry_puff.jpg"
    
    