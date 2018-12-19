import numpy as np
from classifiers.mlp import Classifier_MLP
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
flist.sort()
flist=[flist[0]]
nameNetwork='mlp'+' '+flist[0]

csvfile = open('data/results/'+flist[0]+'.csv', 'w')
filewriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
filewriter.writerow(['Name','num_layers','units', 'Acc1','Acc2','Acc3','Acc4','Acc5','Acc6','Acc7','Acc8','Acc9','Acc10','Avg'])

l=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.25,1.50,1.75,2,3,4]
h_layers=3
for fname in flist:
    scores=[]
    x_train , x_test, y_train, y_test = train_test_splitUCR(fdir,fname)
    f=open("Eye.txt","a")
    a = datetime.datetime.now()
    f.write("\n"+fname+ " " + str(a) +"\n")
    nb_classes = len(np.unique(y_test))
    x_train , x_test, y_train, y_test = preprocessing(x_train , x_test, y_train, y_test)
    input_shape = x_train.shape[1:]
    print(input_shape[0])
    l = [int(np.floor(x*input_shape[0])) for x in l]
    print(l)
    for layers in range(1,h_layers+1):
        for units in l:
            scores=[]
            for i in range(10):
                c =Classifier_MLP(input_shape,nb_classes,units,layers)
                print("\n[" +fname+"]Treinando a rede ",i+1,"° vez")
                hist = c.fit(x_train,y_train,x_test,y_test)
                e = str(len(hist.history['loss']))
                print("Parou em "+e+" epocas")
                score = c.model.evaluate(x_test,y_test)
                scores.append(round(score[1]*100,2))
                print("%d Layers"%(layers))
                print("%d Units"%(units))
                print("%s: %.2f%%"%(c.model.metrics_names[1],score[1]*100))

            avg = round(sum(scores)/len(scores),2)
            print('(Units,Avg)=',units,avg)

            #f.write("AVG Acc %.2f%%:"%(avg))
            #f.write("\n")
            #f.close()
            filewriter.writerow([fname]+[layers]+[units]+scores+[avg])
csvfile.close()

s.sendemail("clemilton.ufam@gmail.com","Treino "+nameNetwork,"Terminou!!")

