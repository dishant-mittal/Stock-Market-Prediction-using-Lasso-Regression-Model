import csv as csv
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
import pylab as pl
from IPython.display import Image, SVG
csv_file = csv.reader(open('second.csv'))
header= csv_file.next()

data = []

for row in csv_file:
	data.append(row)

data = np.array(data)

for i in range(0,3686):
	data[i,0] = data[i,0].replace('-',"")

#print(data)

xtrain = data[1106:3686,0:6].astype(np.float)
ytrain = data[1106:3686,6].astype(np.float)
xtest = data[0:1105,0:6].astype(np.float)
ytest = data[0:1105,6].astype(np.float)

x_temp = []
for i in range(1105,0,-1):
	x_temp.append(i)

x_temp = np.array(x_temp).astype(int)

#rid = linear_model.LinearRegression(normalize = True)
#rid.fit(xtrain,ytrain)

#rid = linear_model.Ridge( alpha = 0.09, normalize = True )
#rid.fit(xtrain,ytrain)


#clf = linear_model.RidgeCV(alphas = [0.01, 0.1, 1.0, 10.0], normalize=True, cv=3)
#clf.fit(xtrain,ytrain)

#print(clf.alpha_)
#print(clf.cv_values)

#rid = ElasticNet(fit_intercept = True, alpha=0.000067, max_iter = 10000, normalize=True)
#rid.fit(xtrain,ytrain)

clf = linear_model.Lasso(alpha = 26)
clf.fit(xtrain,ytrain)

p = map(clf.predict,xtrain)
e = p - ytrain
err = np.sum(e*e)
rmse = np.sqrt(err/len(xtrain))
print('Training prediction:',clf.predict(xtrain))
print('\nRMSE on training: {}'.format(rmse))
'''
pl.scatter(p,ytrain,color='green',alpha=0.2)
x1 = np.linspace(0,240,1000)
y1 = x1
pl.plot(x1,y1,color='black',linewidth=0.5)
pl.xlim(0,240)
pl.ylim(0,240)
pl.xlabel('Predicted output value')
pl.ylabel('Actual output value')
pl.title('Goldman Sachs training set')
pl.savefig('ridge1.png', format='png', dpi=1000)
#pl.show()
'''


p = map(clf.predict,xtest)
e = p - ytest

print("MAPE: ")
print(np.mean(np.abs((ytest- p) / p)) * 100)
'''
pl.plot(x_temp,ytest,color = 'green',alpha = 0.6, label='Actual Output')
pl.plot(x_temp,p,color = 'brown', alpha = 0.6, label='Predicted Output')
pl.xlabel('Number of days')
pl.title('Ridge Regression')
pl.legend(loc='upper right')
pl.ylabel('Output value')
pl.savefig('ridge2.png', format='png', dpi=1000)
#pl.show()
'''


pl.scatter(p,ytest,color='brown',alpha=0.2)
x1 = np.linspace(0,200,1000)
y1 = x1
pl.plot(x1,y1,color='black',linewidth=0.5)
pl.xlim(0,200)
pl.ylim(0,200)
pl.xlabel('Predicted output value')
pl.ylabel('Actual output value')
pl.title('Goldman Sachs test set')
pl.savefig('lasso3.png', format='png', dpi=600)
#pl.show()



err = np.sum(e*e)
rmse = np.sqrt(err/len(xtest))
print('Testing prediction:',clf.predict(xtest))
print('\nRMSE on testing: {}'.format(rmse))

#p = map(rid.predict,xtest)
#err =np.absolute(p - ytest)
#for i in range(0,1503):
	#print(np.absolute(err[i]))
#print(ytest)
#for i in range(0,1503):
#  	err[i] = err[i]/ytest[i]

#print(err)
#var=0
#for i in range(0,1503):
#   	var=err[i]+var
#var=np.sum(err)
#print(var)
#mape=(var/1503)*100
#print(mape)


#print(rid.get_params)

