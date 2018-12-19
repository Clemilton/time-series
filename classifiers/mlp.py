import keras

class Classifier_MLP:
    
    def __init__(self,input_shape,nb_classes,n_units=500,layers=1):
        self.model = self.build_model(input_shape,nb_classes,n_units,layers)
        
    def build_model(self,input_shape,nb_classes,n_units,layers):
        x = keras.layers.Input(input_shape)
        y= keras.layers.Dropout(0.1)(x)
        for i in range(layers):
            if i==0:
                y = keras.layers.Dense(n_units, activation='relu')(x)
            else:
                y = keras.layers.Dense(n_units, activation='relu')(y)
            y = keras.layers.Dropout(0.2)(y)
        
        
        out = keras.layers.Dense(nb_classes, activation='softmax')(y)
        model = keras.models.Model(inputs=[x], outputs=[out])
        optimizer = keras.optimizers.Adadelta() 
        model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                      factor=0.5,
                                      patience=10,
                                      verbose=False,
                                     min_delta=0.0001,
                                     cooldown=3,
                                     min_lr=0)
        early = keras.callbacks.EarlyStopping(monitor='loss',min_delta=0,patience=50,verbose=False)
        
        self.callbacks =[reduce_lr,early]
        
        return model
                                 
    def fit(self,x_train,y_train,x_test,y_test,nb_epochs=1000):
         batch_size = min(int(x_train.shape[0]/10),16)
            
         h =self.model.fit(x_train,y_train,batch_size=batch_size,epochs=nb_epochs,verbose=False,
                        validation_split=0.2,callbacks=self.callbacks)
         return h
                                 