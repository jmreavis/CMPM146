from models.model import Model
from tensorflow.keras import layers, models
import tensorflow as tf

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here

        
        self.model = models.Sequential()
        #keras rescaling layer <- use this
        self.model.add(tf.keras.layers.Rescaling(scale=1./255, offset=0.0))
        #could also change training input
        self.model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dense(categories_count, activation='softmax'))
        pass
    
    def _compile_model(self):
        # Your code goes here

        # self.model.compile(<configuration properties>)
        self.model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        pass
