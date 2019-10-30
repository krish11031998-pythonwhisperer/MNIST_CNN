import keras
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Dense,Flatten
import os
import numpy as np
# import matplotlib.pyplot as plt
import talos as ta
# from __future__ import print_function
import Tuning_CNN as tuning
import finalizing_CNN_design as fcd


class MNIST_data():

	def __init__(self,dataset):
		(train_X,train_Y),(test_X,test_Y) = dataset.load_data()
		self.train_X = train_X
		self.train_Y = train_Y
		self.test_X = test_X
		self.test_Y = test_Y
		self.data_preprocessing()

	def data_preprocessing(self):
		self.train_X = (self.train_X.reshape(-1,28,28,1).astype('float32'))/255
		self.test_X = (self.test_X.reshape(-1,28,28,1).astype('float32'))/255

		self.train_Y = to_categorical(self.train_Y)
		self.test_Y = to_categorical(self.test_Y)


class Model(MNIST_data):

	def __init__(self,dataset,name_json,name_weights,vers=1,params=None):
		super().__init__(dataset)
		self.name_json = name_json
		self.name_weights = name_weights
		self.vers = vers
		if params :
			self.params = params
			self.batch_size = params['batch_size']
		else:
			self.batch_size = 64


	def CNN_model(self):


		if self.vers == 1:
			self.model = Sequential()
			self.model.add(Conv2D(64,(3,3),input_shape=(28,28,1)))
			self.model.add(Activation('relu'))
			self.model.add(MaxPool2D(pool_size=(2,2)))
			self.model.add(Dropout(0.2))

			self.model.add(Conv2D(64,(3,3)))
			self.model.add(Activation('relu'))
			self.model.add(MaxPool2D(pool_size=(2,2)))
			self.model.add(Dropout(0.2))


			self.model.add(Flatten())
			self.model.add(Dense(64))
			#optional Dense layer
			#self.model.add(Activation('relu'))
			self.model.add(Dense(10))
			self.model.add(Activation('softmax'))

			self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

		elif self.vers == 2:

			params = self.params

			dropout_num = params['dropout']
			dense_neuron = params['dense_neuron']


			self.model = Sequential()
			self.model.add(Conv2D(64,(3,3),input_shape=(28,28,1)))
			self.model.add(Activation(params['activation']))
			self.model.add(MaxPool2D(pool_size=(2,2)))
			self.model.add(Dropout(dropout_num))

			self.model.add(Conv2D(64,(3,3),input_shape=(28,28,1)))
			self.model.add(Activation(params['activation']))
			self.model.add(MaxPool2D(pool_size=(2,2)))
			self.model.add(Dropout(dropout_num))


			self.model.add(Flatten())
			self.model.add(Dense(dense_neuron))
			self.model.add(Activation(params['activation']))
			self.model.add(Dense(10))
			self.model.add(Activation('softmax'))

			self.model.compile(loss='categorical_crossentropy',optimizer=params['optimizer'],metrics=['accuracy'])


	def fit_train(self):
		self.model.fit(x=self.train_X,y=self.train_Y,epochs=10,batch_size=self.batch_size)


	def save_classifier(self):
		save_model = self.model.to_json()
		with open(self.name_json,'w') as json_file:
			json_file.write(save_model)
		self.model.save_weights(self.name_weights)
		print("Successfully saved the CNN model")


	def load_classifier(self):
		with open(self.name_json,'r') as json_file:
			saved_model = json_file.read()
		self.model = keras.models.model_from_json(saved_model)
		self.model.load_weights(self.name_weights)
		self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


	def running_classifier(self):
		if os.path.isfile(self.name_json) and os.path.isfile(self.name_weights):
			self.load_classifier()
		else:
			self.CNN_model()
			self.fit_train()
			self.save_classifier()

	def evaluate_model(self):

		test_loss,test_acc = self.model.evaluate(self.test_X,self.test_Y)
		return (test_loss,test_acc)


	def predict(self):

		prediction = self.model.predict(self.test_X)
		print(np.argmax(np.round(prediction[0])))


if __name__=="__main__":
	model = Model(fashion_mnist,name_json='MNIST_Classifier.json',name_weights='MNIST_Weights.h5',vers=1,params=None)
	model.running_classifier()
	model.predict()
	tuning.tune()
	params = fcd.final_params()
	model_2 = Model(fashion_mnist,name_json='MNIST_Classifier_tuned.json',name_weights='MNIST_Weights_tuned.h5',vers=2,params=params)
	model_2.running_classifier()
	model_2.predict()
	test_loss_1,test_acc_1 = model.evaluate_model()
	test_loss_2,test_acc_2 = model_2.evaluate_model()
	print('The model evaluation before and after tuning is as follows:')
	print("Before Tuning : loss = {}, acc= {}".format(test_loss_1,test_acc_1))
	print("After tuning: loss={} , acc={}".format(test_loss_2,test_acc_2))
