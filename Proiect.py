import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# Încarcare date
file = 'REMS_Mars_Dataset.csv'
data = pd.read_csv(file)

#Vizualizare date
df = pd.read_csv('REMS_Mars_Dataset.csv')
df.head()

# Selecție date intrare
X = df[['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Pressure (millibars)']]
X = np.array(X)
X.shape

# Selecție date iesire
y = df['Temperature (C)']
y = np.array(y)
y.shape

# Split pentru setul de antrenare și testare
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)


# Normalizare
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construire model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(90,activation='relu'),
    tf.keras.layers.Dense(4,activation='relu'),
    tf.keras.layers.Dense(1)
])

#Compilare model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mae')

#Antrenarea modelului
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training','Validation'])
plt.show()

#Predictie
y_pred = model.predict(X_test)

print(y_pred[:10])
print(y_test[:10])

#Impartim setul de date
def batch(X, y , batch_size):
    for i in range(0,len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]
batch(X_train, y_train, 32)
batch(X_test, y_test, 32)


# Construire model2
model_2 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(90,activation='relu'),
    tf.keras.layers.Dense(4,activation='relu'),
    tf.keras.layers.Dense(1)
])


#Compilare model
model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mae')

#Antrenarea modelului
history = model_2.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training','Validation'])
plt.show()

#Predictie2
y_pred = model_2.predict(X_test)

#Date prezise
print(y_pred[:10])
#Date reale
print(y_test[:10])