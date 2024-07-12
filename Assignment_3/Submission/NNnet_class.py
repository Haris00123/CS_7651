import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l1_l2
import tensorflow as tf
from keras.callbacks import EarlyStopping

class NNnet:

    '''Custom class using Keras to create Neural Network models, but, remaining compatible with sklearn'''

    def __init__(self,internal_layers=8,epochs=250,units=128,batch_size=32,input_dims=17,internal_act='elu',final_act='softmax',optimizer=Adam,learning_rate=0.001,loss='sparse_categorical_crossentropy',metrics=['accuracy'],l1_reg=0,l2_reg=0,dropout_rate=0,early_stopping_tol=0.001,multi_labels=10):
        '''Class init

        Parameters:
        internal_layers (int): Number of internal layers
        epochs (int): Number of epochs to train for
        batch_size (int): Batch size for model
        input_dims (int): Input dimensions for data
        internal_act (str): Internal activation
        final_act (str): Final layer activation
        optimzer (keras.optimizers): Optimizer
        learning rate (int): Learning rate for model
        lss (str): Loss function
        metrics (list): List of metrics
        l1_reg (int): Integer for L1 regularization
        l2_reg (int): Integer for L2 regularization
        dropout_rate (int): Dropour rate
        eraly_stopping_tol (int): Tolerance for early stopping'''
        self.internal_model=Sequential()

        #Setting parameters
        self.internal_layers=internal_layers
        self.epochs=epochs
        self.units=units
        self.batch_size=batch_size
        self.input_dims=input_dims
        self.internal_act=internal_act
        self.final_act=final_act
        self.optimizer=optimizer
        self.learning_rate=learning_rate
        self.loss=loss
        self.metrics=metrics
        self.l1_reg=l1_reg
        self.l2_reg=l2_reg
        self.dropout_rate=dropout_rate
        self.early_stopping_tol=early_stopping_tol
        self.multi_labels=multi_labels

        if 'F1Score' in metrics:
            self.f1=True
        else:
            self.f1=False
        
        self.build_model()

    def build_model(self):
        '''Build Keras model'''
        self.internal_model=Sequential()
        self.internal_model.add(Dense(units=self.units,activation=self.internal_act,input_shape=(self.input_dims,)))
        for i in range(self.internal_layers):
            self.internal_model.add(Dense(units=self.units,activation=self.internal_act,kernel_regularizer=l1_l2(l1=self.l1_reg,l2=self.l2_reg)))
            if self.dropout_rate>0:
                self.internal_model.add(Dropout(self.dropout_rate))

        self.internal_model.add(Dense(units=self.multi_labels,activation=self.final_act))

        self.optimizer_final=self.optimizer(self.learning_rate)
        self.internal_model.compile(optimizer=self.optimizer_final,loss=self.loss,metrics=self.metrics)  

    def fit(self,X,y,verbose=0,early_stopping=True,validation_data=None):
        '''Fit the kears model
        
        Parameters:
        X (np.array): Features
        y (np.array): Labels
        verbose (int): Parameter for model fit
        early_stopping (bool): If early stopping callback is to be used
        validation_data (list): Validation data'''
        if self.f1:
            y_copy=tf.cast(y,tf.float32)
        else:
            y_copy=y

        #If validation Data
        if validation_data:
            if early_stopping:
                cb=EarlyStopping(monitor='val_loss',min_delta=self.early_stopping_tol,patience=3,verbose=0,start_from_epoch=10)

                if self.f1:
                    validation_data[-1]=tf.cast(validation_data[-1],tf.float32)

            self.internal_model.fit(X,y_copy,epochs=self.epochs,batch_size=self.batch_size,verbose=verbose,callbacks=[cb],validation_data=validation_data)
        else:
            if early_stopping:
                cb=EarlyStopping(monitor='loss',min_delta=self.early_stopping_tol,patience=3,verbose=0)   
            self.internal_model.fit(X,y_copy,epochs=self.epochs,batch_size=self.batch_size,verbose=verbose,callbacks=[cb])

        return self

    def predict(self,X,verbose=0):
        '''Inference function
        
        Parameters:
        X (np.array): Features
        verbose (int): Parameter for model fit

        Returns:
        (np.array): Predictions
        '''
        if self.multi_labels>1:
            return np.argmax(np.round(self.internal_model.predict(X,verbose=verbose)),axis=1)
        else:
            return np.round(self.internal_model.predict(X,verbose=verbose))

    def get_params(self,deep=True):
        parameters = {
            'internal_layers': self.internal_layers,
            'epochs': self.epochs,
            'units': self.units,
            'batch_size': self.batch_size,
            'input_dims': self.input_dims,
            'internal_act': self.internal_act,
            'final_act': self.final_act,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'loss': self.loss,
            'metrics': self.metrics,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'dropout_rate': self.dropout_rate
        }
        return parameters

    def copy(self):
        '''Copy Model
        
        Returns (NNnet)'''
        return NNnet(**self.get_params())

    def set_params(self,**parameters):
        '''Set parameters
        
        Parameters:
        parameters (dict): Parameters to change
        '''
        for parameter,value in parameters.items():
            setattr(self, parameter, value)
        self.build_model()
        return self