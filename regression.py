from sklearn import linear_model
import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/Anupam Suman/Desktop/Summer Project/training.csv")
data2 = pd.read_csv("C:/Users/Anupam Suman/Desktop/Summer Project/test.csv")
data = np.array(data)

y = data[:,12]
SiO2 = data[:,0]
SO3 = data[:,1]
Fe2O3 = data[:,2]
Al2O3 = data[:,3]
Cao = data[:,4]
C = data[:,5]
Na2O = data[:,6]
MgO = data[:,7]
Cl = data[:,8]
K20 = data[:,9]
TiO2= data[:,10]
Sro = data[:,11]
X = [list(SiO2), list(SO3), list(Fe2O3), list(Al2O3), list(Cao), list(C), list(Na2O), list(MgO), list(Cl), list(K20), list(TiO2), list(Sro)]
X = np.round(X,2)
np.set_printoptions(formatter={'float_kind':'{:f}'.format})


print(X)
print(y)

regr = linear_model.LinearRegression()
regr.fit(np.array(X).transpose(), y)

data2 = np.array(data2)
print(data2)


p_id = data2[:,0]
SiO2_ = data2[:,1]
SO3_ = data2[:,2]
Fe2O3_ = data2[:,3]
Al2O3_ = data2[:,4]
Cao_ = data2[:,5]
C_ = data2[:,6]
Na2O_ = data[:,7]
MgO_ = data[:,8]
Cl_ = data[:,9]
K20_ = data[:,10]
TiO2_= data[:,11]
SrO_ = data[:,12]

X_new = [list(SiO2_), list(SO3_), list(Fe2O3_), list(Al2O3_), list(Cao_), list(C_), list(Na2O_), list(MgO_), list(Cl_), list(K20_), list(TiO2_), list(SrO_)]
X_new = np.round(X,2)
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
print(X_new)
Y_new = regr.predict(np.array(X_new).transpose())
print(Y_new)
a = np.array(p_id)
b = np.array(Y_new)
p = [a,b]
print(p)
pd.DataFrame(p).transpose().to_csv("C:/Users/Anupam Suman/Desktop/Summer Project/my_solution.csv", index = 0)

