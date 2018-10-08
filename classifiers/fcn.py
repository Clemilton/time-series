import keras

class Classifier_FCN:
    
    def __init__(self,input_shape,nb_classes):
        self.model = self.build_model(input_shape,nb_classes)
        
    def build_model(self,input_shape,nb_classes):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.normalization.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)
        
        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.normalization.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
        conv3 = keras.layers.normalization.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)
        
        gap_layer = keras.layers.pooling.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
            metrics=['accuracy'])
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                             factor=0.5,
                             patience=10,
                             verbose=1,
                             mode='auto',
                             min_delta=0.0001, cooldown=3, min_lr=0) 
        early = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=30,verbose=0)


        self.callbacks = [reduce_lr,early]

        return model

    def fit(self,x_train,y_train,x_test,y_test,nb_epochs=1000):
         batch_size = min(int(x_train.shape[0]/10),16)
            
         h =self.model.fit(x_train,y_train,batch_size=batch_size,epochs=nb_epochs,verbose=False,
                        validation_split=0.2,callbacks=self.callbacks)
         return h
                                 