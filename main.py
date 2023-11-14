import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from keras.layers import Dense
from tensorflow.keras.models import Sequential

df=pd.read_csv('Cancer_Data.csv')
df.head()
df1=df.drop('Unnamed: 32',axis=1)
df1=df1.drop('id',axis=1)

df1

df1['diagnosis'].replace('M',1,inplace=True)
df1['diagnosis'].replace('B',0,inplace= True)
df1[df1.select_dtypes(np.float64).columns] = df1.select_dtypes(np.float64).astype(np.float32)
df1.info()
x = df1.iloc[:, 1:].values
y = df1.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 110)

# Model 1
model1 = Sequential()
model1.add(Dense(64, activation='relu'))
model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adam')
history = model1.fit(X_train,y_train,epochs=100,verbose=1, batch_size=32, validation_data=(X_test, y_test))
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# Model 2
model2 = Sequential()
model2.add(Dense(64, activation='relu'))
model2.add(Dense(128, activation='relu'))
model2.add(Dense(1,activation='sigmoid'))
model2.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adam')
history2 = model2.fit(X_train,y_train,epochs=100,verbose=1, batch_size=32, validation_data=(X_test, y_test))

plt.plot(history2.history['binary_accuracy'])
plt.plot(history2.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Model 3
model3 = Sequential()
model3.add(Dense(1024, activation='relu'))
model3.add(Dense(512, activation='relu'))
model3.add(Dense(128, activation='relu'))
model3.add(Dense(1,activation='sigmoid'))
model3.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adam')
history3 = model3.fit(X_train,y_train,epochs=100,verbose=1, batch_size=32, validation_data=(X_test, y_test))

plt.plot(history3.history['binary_accuracy'])
plt.plot(history3.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
