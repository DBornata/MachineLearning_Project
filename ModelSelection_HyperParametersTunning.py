import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Question 1.1 Dana Bornata 1200284
data = pd.read_csv('data_reg.csv')
trainingset = data[:120]
validationset = data[120:160]
testing_set = data[160:]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(trainingset['x1'], trainingset['x2'], trainingset['y'], c='r', marker='o', label='Training Set')
ax.scatter(validationset['x1'], validationset['x2'], validationset['y'], c='g', marker='s', label='Validation Set')
ax.scatter(testing_set['x1'], testing_set['x2'], testing_set['y'], c='b', marker='^', label='Testing Set')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend()
plt.show()

# Question 1.2 Dana Bornata 1200284

Xtrainingset = trainingset[['x1', 'x2']].values
Ytrainingset = trainingset['y'].values
Xvalidationset = validationset[['x1', 'x2']].values
Yvalidationset = validationset['y'].values
COUNT_Degree = np.arange(1, 11)
trainingset_Errors = []
validationset_Errors = []

for degree in COUNT_Degree:
    polynomial = PolynomialFeatures(degree=degree)
    Xtrainingset_polynomial = polynomial.fit_transform(Xtrainingset)
    validationset_polynomial = polynomial.transform(Xvalidationset)
    model = LinearRegression()
    model.fit(Xtrainingset_polynomial, Ytrainingset)
    Ytraining_Predict = model.predict(Xtrainingset_polynomial)
    Yvalidation_Predict = model.predict(validationset_polynomial)
    trainingError = mean_squared_error(Ytrainingset, Ytraining_Predict)
    validationError = mean_squared_error(Yvalidationset, Yvalidation_Predict)
    trainingset_Errors.append(trainingError)
    validationset_Errors.append(validationError)
    Result_figure = plt.figure(figsize=(6, 6))
    ax_surface = Result_figure.add_subplot(111, projection='3d')
    ax_surface.scatter(Xtrainingset[:, 0], Xtrainingset[:, 1], Ytrainingset, c='g', marker='o', label='Training Set')
    x1Range = np.linspace(min(Xtrainingset[:, 0]), max(Xtrainingset[:, 0]), 100)
    x2Range = np.linspace(min(Xtrainingset[:, 1]), max(Xtrainingset[:, 1]), 100)
    x1Mesh, x2_mesh = np.meshgrid(x1Range, x2Range)
    X_mesh = np.c_[x1Mesh.ravel(), x2_mesh.ravel()]
    X_mesh_poly = polynomial.transform(X_mesh)
    y_mesh_pred = model.predict(X_mesh_poly)
    ax_surface.plot_surface(x1Mesh, x2_mesh, y_mesh_pred.reshape(x1Mesh.shape), alpha=0.5, cmap='viridis', label=f'Degree {degree}')
    ax_surface.set_xlabel('X1')
    ax_surface.set_ylabel('X2')
    ax_surface.set_zlabel('Y')
    ax_surface.legend()

    plt.show()

# Plot the validation error vs polynomial degree curve
plt.plot(COUNT_Degree, validationset_Errors, marker='o')
plt.title('validation error vs polynomial degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.show()




# Question 1.3 Dana Bornata 1200284
ridge_Option = [0.001, 0.005, 0.01, 0.1, 10]
ridge_validationErrors = []
for alpha in ridge_Option:
    Polynomial = PolynomialFeatures(degree=8)
    X_train_Polynomial = Polynomial.fit_transform(Xtrainingset)
    Xvalidation_Polynomial = Polynomial.transform(Xvalidationset)
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train_Polynomial, Ytrainingset)
    yvalidation_pred_ridge = ridge_model.predict(Xvalidation_Polynomial)
    val_error_ridge = mean_squared_error(Yvalidationset, yvalidation_pred_ridge)
    ridge_validationErrors.append(val_error_ridge)

best_value = ridge_Option[np.argmin(ridge_validationErrors)]
best_ridge_model = Ridge(alpha=best_value)
best_ridge_model.fit(X_train_Polynomial, Ytrainingset)
y_val_pred_best_ridge = best_ridge_model.predict(Xvalidation_Polynomial)

plt.figure(figsize=(8, 6))
plt.plot(ridge_Option, ridge_validationErrors, marker='o', label='Ridge Regression')
plt.axvline(x=best_value, color='r', linestyle='--', label=f'Best Value: {best_value:.3f}')
plt.xscale('log')
plt.title('MSE on the validation vs the regularization parameter')
plt.xlabel('regularization parameter')
plt.ylabel('MSE on the validation Set')
plt.legend()
plt.show()
