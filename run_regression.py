import datasets
import regression
import importlib
importlib.reload(regression)

X,Y = datasets.load_linear_example1()
model = regression.LinearRegression()
model.fit(X,Y)

# ver.1
print(model.x)

# ver.2
print(model.theta)

# ver.3
print(model.predict(X))
