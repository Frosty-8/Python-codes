import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Users/Sarthak/OneDrive/Documents/Desktop/python/ml/Housing.csv")

data=pd.DataFrame(df)

print(data)

X=data[['bedrooms','bathrooms','area']]
y=data['price']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

model = LR()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Mean squared error : ", metrics.mean_squared_error(y_test,y_pred))
print("\nR2 score : " , metrics.r2_score(y_test,y_pred))
print(f"\nCoefficients : {model.coef_}")
print(f"\nIntercept : {model.intercept_}")

plt.scatter(y_test,y_pred)
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.plot(y_test,y_test,color='red')
plt.show()