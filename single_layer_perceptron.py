import numpy as np
from tensorflow import keras 
import matplotlib.pyplot as plt 
import tensorflow as tf
#from tensorFlow.keras.models import Sequential
#from tensorFlow.keras.layers import Dense

training_data=np.array([[0,0],[0,1],[1,0],[1,1]])
labels=np.array([0,0,0,1])
model=keras.Sequential([
    keras.layers.Dense(1,input_shape=(2,),activation='sigmoid')

])
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history=model.fit(training_data,labels,epochs=1000,verbose=0)
print("testing the perceptron")
for input_data in training_data:
    raw_prediction=model.predict(inp-+ut_data.reshape(1,-1))
    prediction_value=raw_prediction[0][0]
    if prediction_value > 0.5:
        prediction=1
    else:
        prediction=0
    print(f"input {input} ,raw output {prediction_value:.4f}, output {prediction}")

