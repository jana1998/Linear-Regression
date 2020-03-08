import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import datasets, linear_model#DATASETS HAVE EXAMPLE DATASETS ,LINEAR_MODEL CONTAINS THE ALGORITHM
from sklearn.metrics import mean_squared_error, r2_score #SKLEARN.METRICS=CONFUSION MATRIX,MEAN AND R2 USED FOR TALLYING AFTER TRAINING THE MODEL
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:,np.newaxis,2]#DATA FROM ALL THE ROWS THEN CONVERTING TO COLUMN
print(diabetes_X)
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-30]
diabetes_y_train = diabetes.target[:-30]


# Split the targets into training/testing sets
diabetes_X_test = diabetes_X[-30:]
diabetes_y_test = diabetes.target[-30:]
#len(diabetes_X_test)

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='blue')
#plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3);
plt.xticks(()) #A list of positions at which ticks should be placed. You can pass an empty list to disable xticks.
plt.yticks(())
plt.show()

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)
diabetes_y_pred

pd.DataFrame(diabetes_y_test,diabetes_y_pred)


# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
