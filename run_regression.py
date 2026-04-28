import datasets
import regression
import importlib
importlib.reload(regression)

X,Y = datasets.load_linear_example1()

# ver.1
model = regression.LinearRegression()
print(model.x)

# ver.2
model = regression.LinearRegression()
model.fit(X,Y)
print(model.theta)
