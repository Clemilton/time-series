import keras

class Classifier_RESNET:
    
    def __init__(self,input_shape,nb_classes):
        self.model = self.build_model(input_shape,nb_classes)
    
    def build_resnet(self,input_shape, n_feature_maps, nb_classes):
        print('Construindo a resnet')
        x = keras.layers.Input(shape=(input_shape))
        conv_x = keras.layers.normalization.BatchNormalization()(x)
        conv_x = keras.layers.Conv2D(n_feature_maps, (8,1), padding='same')(conv_x)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_y = keras.layers.Conv2D(n_feature_maps, (5, 1), padding='same')(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_z = keras.layers.Conv2D(n_feature_maps, (3, 1), padding='same')(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        is_expand_channels = not (input_shape[-1] == n_feature_maps)
        if is_expand_channels:
            shortcut_y = keras.layers.Conv2D(n_feature_maps, (1, 1),padding='same')(x)
            shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
        else:
            shortcut_y = keras.layers.normalization.BatchNormalization()(x)

        y = keras.layers.add([shortcut_y, conv_z])

        y = keras.layers.Activation('relu')(y)

        x1 = y
        conv_x = keras.layers.Conv2D(n_feature_maps*2, (8,1), padding='same')(x1)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv2D(n_feature_maps*2, (5, 1), padding='same')(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv2D(n_feature_maps*2, (3, 1), padding='same')(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
        if is_expand_channels:
            shortcut_y = keras.layers.Conv2D(n_feature_maps*2, (1, 1),padding='same')(x1)
            shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
        else:
            shortcut_y = keras.layers.normalization.BatchNormalization()(x1)
        y = keras.layers.add([shortcut_y, conv_z])
        y = keras.layers.Activation('relu')(y)

        x1 = y
        conv_x = keras.layers.Conv2D(n_feature_maps*2, (8,1), padding='same')(x1)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv2D(n_feature_maps*2, (5, 1), padding='same')(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv2D(n_feature_maps*2, (3, 1), padding='same')(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
        if is_expand_channels:
            shortcut_y = keras.layers.Conv2D(n_feature_maps*2, (1, 1),padding='same')(x1)
            shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
        else:
            shortcut_y = keras.layers.normalization.BatchNormalization()(x1)
        y = keras.layers.add([shortcut_y, conv_z])
        y = keras.layers.Activation('relu')(y)

        full = keras.layers.pooling.GlobalAveragePooling2D()(y)   
        out = keras.layers.Dense(nb_classes, activation='softmax')(full)
        return x, out
    
    def build_model(self,input_shape,nb_classes):
        x,y = self.build_resnet(input_shape, 64, nb_classes)
        model = keras.models.Model(inputs=x, outputs=y)

        model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
            metrics=['accuracy'])
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                             factor=0.5,
                             patience=10,
                             verbose=0,
                             mode='auto',
                             min_delta=0.0001, cooldown=3, min_lr=0) 
        early = keras.callbacks.EarlyStopping(monitor='loss',min_delta=0,patience=50,verbose=0)


        self.callbacks = [reduce_lr,early]

        return model

    def fit(self,x_train,y_train,x_test,y_test,nb_epochs=1000):
         batch_size = min(int(x_train.shape[0]/10),16)
            
         h =self.model.fit(x_train,y_train,batch_size=batch_size,epochs=nb_epochs,verbose=0,
                        validation_split=0.2,callbacks=self.callbacks)
         return h
                                 
