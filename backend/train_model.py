import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv(r"C:\Users\krish\OneDrive\Desktop\AutoPredict\backend\car_data.csv")
df.head()
df.info()
df.isnull().sum()
encoder=LabelEncoder()
df['Fuel_Type']=encoder.fit_transform(df['Fuel_Type'])
df['Seller_Type']=encoder.fit_transform(df['Seller_Type'])
df['Transmission']=encoder.fit_transform(df['Transmission'])
df.head()
x=df[['Year','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
y=df[['Selling_Price']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model=RandomForestRegressor(n_estimators=100,random_state=42,max_depth=5,min_samples_split=4)
model.fit(x_train,y_train)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)    
print("Model trained and saved as model.pkl")
