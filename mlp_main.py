import numpy as np
from classifiers.mlp import Classifier_MLP
import os
from utils.utils import train_test_splitUCR,preprocessing
import send_email as s
import csv


fdir='data/ucr/'
flist = [os.path.join(fdir, o) for o in os.listdir(fdir) 
                    if os.path.isdir(os.path.join(fdir,o))]
for i in range(len(flist)):
    flist[i]=flist[i][9:]
    
    
csvfile = open('mlp.csv', 'w')
filewriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
filewriter.writerow(['Name', 'Acc1','Acc2','Acc3','Acc4','Acc5','Acc6','Acc7','Acc8','Acc9','Acc10','Avg'])
flist.sort()
flist = flist[:25]
print(flist)


for fname in flist:
    scores=[]
    x_train , x_test, y_train, y_test = train_test_splitUCR(fdir,fname)
    nb_classes = len(np.unique(y_test))
    x_train , x_test, y_train, y_test = preprocessing(x_train , x_test, y_train, y_test)
    input_shape = x_train.shape[1:]
    for i in range(10):
        c =Classifier_MLP(input_shape,nb_classes)
        print("\n[" +fname+"]Treinando a rede ",i+1,"° vez")
        hist = c.fit(x_train,y_train,x_test,y_test)
        e = str(len(hist.history['loss']))
        print("Parou em "+e+" epocas")

        score = c.model.evaluate(x_test,y_test)
        scores.append(round(score[1]*100,2))
        print("%s: %.2f%%"%(c.model.metrics_names[1],score[1]*100))

    avg = round(sum(scores)/len(scores),2)
    print(avg)
    filewriter.writerow([fname]+scores+[avg])
csvfile.close()

nameNetwork='mlp'
##s.sendemail("clemilton.ufam@gmail.com","Treino "+nameNetwork,"Terminou!!")

