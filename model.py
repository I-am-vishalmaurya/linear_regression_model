import numpy as np
import pandas as pd
import pickle

data = pd.read_csv("student_data.csv")

X = data[['Hours']].values
y = data['Scores'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

pred = reg.predict(X_test)



pickle.dump(reg, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[9.25]]))