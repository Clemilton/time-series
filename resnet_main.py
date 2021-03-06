import numpy as np
from classifiers.resnet import Classifier_RESNET
import os
from utils.utils import train_test_splitUCR,preprocessing
import send_email as s
import csv
import datetime

fdir='data/ucr/'
flist = [os.path.join(fdir, o) for o in os.listdir(fdir) 
                    if os.path.isdir(os.path.join(fdir,o))]
for i in range(len(flist)):
    flist[i]=flist[i][9:]
    
    
csvfile = open('resnet.csv', 'w')
filewriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
filewriter.writerow(['Name', 'Acc1','Acc2','Acc3','Acc4','Acc5','Acc6','Acc7','Acc8','Acc9','Acc10','Avg'])
flist.sort()
flist=flist[:25]

cont=1
for fname in flist:
    scores=[]
    x_train , x_test, y_train, y_test = train_test_splitUCR(fdir,fname)
    f=open("Eye.txt","a")
    a = datetime.datetime.now()
    f.write("\n"+str(cont)+" "+fname+ " " + str(a) +"\n")
    nb_classes = len(np.unique(y_test))
    x_train , x_test, y_train, y_test = preprocessing(x_train , x_test, y_train, y_test)
    x_train = x_train.reshape(x_train.shape +(1,1))
    x_test = x_test.reshape(x_test.shape + (1,1))
    input_shape = x_train.shape[1:]
    for i in range(10):
        c =Classifier_RESNET(input_shape,nb_classes)
        print("\n[" +fname+"]Treinando a rede ",i+1,"° vez")
        hist = c.fit(x_train,y_train,x_test,y_test)
        e = str(len(hist.history['loss']))
        print("Parou em "+e+" epocas")

        score = c.model.evaluate(x_test,y_test)
        scores.append(round(score[1]*100,2))
        print("%s: %.2f%%"%(c.model.metrics_names[1],score[1]*100))

    avg = round(sum(scores)/len(scores),2)
    print(avg)
    f.write("AVG Acc %.2f%%:"%(avg))
    f.write("\n")
    f.close()
    cont=cont+1
    filewriter.writerow([fname]+scores+[avg])
csvfile.close()

nameNetwork='resnet'
s.sendemail("clemilton.ufam@gmail.com","Treino "+nameNetwork,"Terminou!!")
