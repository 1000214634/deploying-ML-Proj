# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import numpy as np

#loaded the model
loaded_model=pickle.load(open('E:\Data Science\Machine Learning\ML Projs\Diabetes/trained_model.sav','rb'))

input_data=(10,168,74,0,0,38,0.537,34)

#Changing the input_data to a numpy array
input_data_as_numpy_array=np.asarray(input_data) #The numpy.asarray() function is used to convert n given input to an array.

#reshape the array as we are predicting the one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)


prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print('The person is Non diabetic')
else:
    print('The person is diabetic')