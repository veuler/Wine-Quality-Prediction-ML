from keras.models import Sequential
from keras.layers import Dense
import numpy,sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn import metrics
import matplotlib.pyplot as plt

def fprint(lst1,lst2, toplam):
    say = 0
    for x,y in zip(lst1,lst2):
        print(f"Tahmin edilen kalite: {x} -----> Gerçek kalite: {y}")
        if x == y:
            say += 1
    print(f"\nToplam tahmin sayısı: {toplam}, Doğru tahmin sayısı: {say}, Başarı yüzdesi: {say * 100 / toplam}%")

def ffprint(lst1,lst2, toplam):
    say = 0
    for x,y in zip(lst1,lst2):
        print(f"Tahmin edilen kalite: {x} -----> Gerçek kalite: {y}")
        if round(x) == y:
            say += 1
    print(f"\nToplam tahmin sayısı: {toplam}, Doğru tahmin sayısı: {say}, Başarı yüzdesi: {say * 100 / toplam}%")


dataset = pd.read_csv(r'files\winequality-red.csv', sep=";")

X = dataset[['volatile acidity','citric acid','chlorides','sulphates',
             'alcohol','free sulfur dioxide','total sulfur dioxide','density']]

y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size = 0.34, random_state = 45)

clf1 = SVC(kernel = 'linear')
clf2 = SVC(kernel = 'rbf')
clf3 = SVC(kernel = 'poly')
clf4 = SVC(kernel = 'sigmoid')

rlf1 = SVR(kernel = 'linear')
rlf2 = SVR(kernel = 'rbf')
rlf3 = SVR(kernel = 'poly')
rlf4 = SVR(kernel = 'sigmoid')

clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)
clf4.fit(X_train, y_train)

rlf1.fit(X_train, y_train)
rlf2.fit(X_train, y_train)
rlf3.fit(X_train, y_train)
rlf4.fit(X_train, y_train)

cPredict1 = clf1.predict(X_test)
cPredict2 = clf2.predict(X_test)
cPredict3 = clf3.predict(X_test)
cPredict4 = clf4.predict(X_test)

rPredict1 = rlf1.predict(X_test)
rPredict2 = rlf2.predict(X_test)
rPredict3 = rlf3.predict(X_test)
rPredict4 = rlf4.predict(X_test)

print("------- Support Vector CLASSIFICATIONS -------")
print("Linear: %{}".format(metrics.accuracy_score(y_test,cPredict1) * 100))
print("RBF: %{}".format(metrics.accuracy_score(y_test,cPredict2) * 100))
print("Polynomial: %{}".format(metrics.accuracy_score(y_test,cPredict3) * 100))
print("Sigmoid: %{}".format(metrics.accuracy_score(y_test,cPredict4) * 100))

fprint(cPredict1,y_test, len(y_test))
ffprint(rPredict1,y_test,len(y_test))

print("#"*100)
print(cPredict1)
print(rPredict1)


##############################################################################################################


print("#"*100)
print(f"Total TRAIN examples: {len(X_train)}")
print(f"Total TEST examples: {len(X_test)}")

model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adagrad', metrics=['accuracy'])

model.fit(X, y, epochs=150, batch_size=30, verbose=0)

dongu = 0
while dongu < 2:

    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy: %.2f, LOSS: %f' % (accuracy*100,loss))

    predictions = model.predict(X_test)

    say = 0
    sayc = 0
    for x,y in zip(predictions,y_test):
        if numpy.round(x) == y:
            say += 1
        sayc += 1
        print(f"{sayc} numaralı şarap kalitesi tahmini: {numpy.round_(x)} - Gerçek kalite: {y}")

    print(say, say * 100 / len(y_test))

    with open(r'C:\Users\smart\Desktop\TEZ\mseTXT.txt','a+') as ff:
        ff.write("{}\n".format([loss,accuracy*100,say]))

    plt.figure()
    plt.plot(predictions,y_test,'bo')

    plt.show()
    model.save(r'C:\Users\smart\Desktop\TEZ\modelSarapLOG.h5')

    dongu += 1

