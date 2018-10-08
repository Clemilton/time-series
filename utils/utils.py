import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def train_test_splitUCR(fdir,fname):
    x_train, y_train = readucr(fdir+fname+'/'+fname+'_TRAIN')
    x_test, y_test = readucr(fdir+fname+'/'+fname+'_TEST')
    ##x = np.concatenate((x_train,x_test),axis=0)
    ##y = np.concatenate((y_train,y_test),axis=0)
    ##nb_classes = len(np.unique(y))
    ##x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    ##while(len(np.unique(y_test))!=nb_classes):
    ##    x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    return x_train , x_test, y_train, y_test

def preprocessing(x_train , x_test, y_train, y_test):    
    nb_classes = len(np.unique(y_test))
    ##Diminuir as labels em 1 unidade
    y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
    y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)

    ##One hot encoded
    y_train = np_utils.to_categorical(y_train,nb_classes)
    y_test = np_utils.to_categorical(y_test,nb_classes)

    #normalizacao dados
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean)/(x_train_std)
    x_test = (x_test - x_train_mean)/(x_train_std)
    
    return x_train , x_test, y_train, y_test