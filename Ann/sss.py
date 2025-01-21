import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

data = pd.read_csv('stocks.csv')

data['Next_Open'] = data['Open'].shift(-1)
data['Next_Volume'] = data['Volume'].shift(-1)

data = data.dropna()

X = data[['Open','High','Low','Close','Volume']].values
y = data[['Next_Open','Next_Volume']].values

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model=Sequential([
    Dense(64,activation='relu',input_dim=X_train.shape[1]),
    Dense(32,activation='relu'),
    Dense(2,activation='linear')
])

model.compile(optimizer='adam',loss='mse',metrics=['mae'])

history = model.fit(X_train,y_train,epochs=50,batch_size=1,validation_split=0,verbose=1)

predictions = model.predict(X_test)

predicted_values = scaler_y.inverse_transform(predictions)

for i in range(len(predicted_values)):
    actual_values = scaler_y.inverse_transform([y_test[i]])  # Ensure the input is a 2D array
    print(f"Predicted : {predicted_values[i]}, Actual  : {actual_values}")

print(f"Number of predictions: {len(predicted_values)}")
print(f"Number of actual values: {len(y_test)}")

# model.save('stock_price_predictor.h5')