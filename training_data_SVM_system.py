import numpy as np
import random as rand
from sklearn import svm, metrics

#citire date

file=open("bands.data",'r')
B=file.readlines()
#print(B)

#lista y si impart 75% train, 25% test

y=[]
y_train=np.zeros(384)   
y_test=np.zeros(128)    

#lista x si impart 75% train, 25% test

x=[]
x_train=np.zeros((384,39))
x_test=np.zeros((128,39))

#creez un vector

vector=[]
for lin in B:
    vector.append(lin.split(','))
  #  print(vector)
  
# inlocuire toate nebuniile
    
for i in range(512):
    if vector[i][39]=='1\n':
        vector[i][39]='1'
    else:
        vector[i][39]='0'
    if vector[i][38]=='?':
        vector[i][38]=rand.randrange(80,120,1)
    if vector[i][37]=='?':
        vector[i][37]=round(rand.uniform(70,120), 2)
    if vector[i][36]=='?':
        vector[i][36]=rand.randrange(20,50,1)
    if vector[i][35]=='?':
        vector[i][35]=rand.randrange(15,120,1)
    if vector[i][34]=='?':
        vector[i][34]=round(rand.uniform(0,3), 1)
    if vector[i][33]=='?':
        vector[i][33]=round(rand.uniform(0,4), 1)
    if vector[i][32]=='?':
        vector[i][32]=rand.randrange(0,10,1)
    if vector[i][31]=='?':
        vector[i][31]=rand.randrange(0,10,1)
    if vector[i][30]=='?':
        vector[i][30]=round(rand.uniform(0,100), 1)
    if vector[i][29]=='?':
        vector[i][29]=round(rand.uniform(0,100), 1)
    if vector[i][28]=='?':
        vector[i][28]=rand.randrange(0,4000,50)
    if vector[i][27]=='?':
        vector[i][27]=round(rand.uniform(0,100), 1)
    if vector[i][26]=='?':
        vector[i][26]=rand.randrange(10,75,1)
    if vector[i][25]=='?':
        vector[i][25]=round(rand.uniform(0,2), 4)
    if vector[i][24]=='?':
        vector[i][24]=rand.randrange(5,120,1)
    if vector[i][23]=='?':
        vector[i][23]=round(rand.uniform(5,30), 1)
    if vector[i][22]=='?':
        vector[i][22]=round(rand.uniform(0,1), 3)
    if vector[i][21]=='?':
        vector[i][21]=rand.randrange(0,100,1)
    if vector[i][20]=='?':
        vector[i][20]=rand.randrange(0,100,1)
    if vector[i][19]=='?':
        vector[i][19]=rand.randrange(1910,1911,1)
    if vector[i][18]=='?':
        vector[i][18]=rand.randrange(0,4,1)
    if vector[i][17]=='?':
        vector[i][17]=rand.choice(['0', '1', '2'])
    if vector[i][16]=='?':
        vector[i][16]=rand.randrange(1,10,1)
    if vector[i][15]=='?':
        vector[i][15]=rand.choice(['821', '802', '813','824','815','816','827','828'])
    if vector[i][14]=='?':
        vector[i][14]=rand.choice(['0', '1', '2','3'])
    if vector[i][13]=='?':
        vector[i][13]=rand.choice(['0', '1'])
    if vector[i][12]=='?':
        vector[i][12]=rand.choice(['0', '1', '2','3'])
    if vector[i][11]=='?':
        vector[i][11]=rand.choice(['0', '1'])   
    if vector[i][10]=='?':
        vector[i][10]=rand.choice(['0', '1','3'])
    if vector[i][9]=='?':
        vector[i][9]=rand.choice(['0', '1','2'])
    if vector[i][8]=='?':
        vector[i][8]=rand.choice(['0', '1','2'])
    if vector[i][7]=='?':
        vector[i][7]=rand.choice(['0', '1','2'])
    if vector[i][6]=='?':
        vector[i][6]=rand.choice(['0', '1'])  
    if vector[i][5]=='?':
        vector[i][5]=rand.choice(['0', '1'])  
    if vector[i][4]=='?':
        vector[i][4]=rand.choice(['0', '1'])  
    
#se creaza y ca fiind ultima coloana
        
    y.append(vector[i][39])        
        
#print(vector)  
    
#se creaza x fara primele 4 coloane  si fara ultima   
for i in range(512):
        x.append(vector[i])   
for j in x: 
    del j[39]
    del j[3]
    del j[2]
    del j[1]
    del j[0]
#print(x)
#print(y)

    
#print(y)
#print(len(y))

x=np.asarray(x)
y=np.asarray(y)

#print(x)
#print(y)

#Impartire train/test
for i in range(384):
    for j in range(35):
        x_train[i][j]=x[i][j]
    y_train[i]=y[i]
for i in range(384,512):
    for j in range(35):
        x_test[i-384][j]=x[i][j]
    y_test[i-384]=y[i]
    

#print(x_train)
#print(x_test)

Cost=[pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(2,0),pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6),pow(2,7)]
#vector cu diferite valori ale lui Cost

for i in range(len(Cost)):

    clf=svm.SVC(kernel='linear',C=Cost[i],gamma=1)
    #se antreneaza sistemul
    clf.fit(x_train,y_train)   
    #se testeaza sistemul
    predictie=clf.predict(x_test)
    acuratete=metrics.accuracy_score(y_test, predictie) #acuratetea
    print("Acurat pt costul "+str(Cost[i])+" este:"+str(acuratete))



