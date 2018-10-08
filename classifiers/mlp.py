import keras

class Classifier_MLP:
    
    def __init__(self,input_shape,nb_classes):
        self.model = self.build_model(input_shape,nb_classes)
        
    def build_model(self,input_shape,nb_classes):
        x = keras.layers.Input(input_shape)
        y= keras.layers.Dropout(0.1)(x)
        y = keras.layers.Dense(500, activation='relu')(x)
        y = keras.layers.Dropout(0.2)(y)
        y = keras.layers.Dense(500, activation='relu')(y)
        y = keras.layers.Dropout(0.2)(y)
        y = keras.layers.Dense(500, activation = 'relu')(y)
        y = keras.layers.Dropout(0.3)(y)
        
        out = keras.layers.Dense(nb_classes, activation='softmax')(y)
        model = keras.models.Model(inputs=[x], outputs=[out])
        optimizer = keras.optimizers.Adadelta() 
        model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                      factor=0.5,
                                      patience=100,
                                      verbose=True,
                                     min_delta=0.0001,
                                     cooldown=3,
                                     min_lr=0.01)
        early = keras.callbacks.EarlyStopping(monitor='loss',min_delta=0,patience=100,verbose=False)
        
        self.callbacks =[reduce_lr]
        
        return model
                                 
    def fit(self,x_train,y_train,x_test,y_test,nb_epochs=1000):
         batch_size = min(int(x_train.shape[0]/10),16)
            
         h =self.model.fit(x_train,y_train,batch_size=batch_size,epochs=nb_epochs,verbose=True,
                        validation_split=0.2,callbacks=self.callbacks)
         return h
                                 