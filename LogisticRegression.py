import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

 # Q2-1 Dana 1200284

training_data = pd.read_csv('train_cls.csv')# Load the training data
testing_data = pd.read_csv('test_cls.csv') # Load the testing data
Xtraing = training_data[['x1', 'x2']].values
Ytraing = training_data['class'].values
Xtesting = testing_data [['x1', 'x2']].values
Ytesting = testing_data ['class'].values
# Initialize Logistic Regression model with linear decision boundary
linear_model = LogisticRegression()
linear_model.fit(Xtraing, Ytraing)
plt.figure(figsize=(6, 6))
plt.scatter(Xtraing[:, 0],Xtraing[:, 1], c=Ytraing, edgecolors='k', marker='o', label='Training Set')
plt.xlabel('X1')
plt.ylabel('X2')
# Plot decision boundary
a1 = plt.gca()
xlim = a1.get_xlim()
ylim = a1.get_ylim()
x, y = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
Z = linear_model.decision_function(np.c_[x.ravel(), y.ravel()])
Z = Z.reshape(x.shape)
# Plot decision boundary
plt.contour(x, y, Z, colors='red', levels=[0], alpha=0.5, linestyles=['-'])
# Add legend for class labels
legend_labels = ['C1', 'C2']
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)]
plt.legend(handles=handles, labels=legend_labels)
plt.title('Logistic regression model with a linear decision boundary')
plt.show()
# Compute training accuracy
train_preds = linear_model.predict(Xtraing)
train_accuracy = accuracy_score(Ytraing, train_preds)
print(f'Training Accuracy (Linear): {train_accuracy}')
# Compute testing accuracy
test_preds = linear_model.predict(Xtraing)
test_accuracy = accuracy_score(Ytraing, test_preds)
print(f'Testing Accuracy (Linear): {test_accuracy}')

#Q2-2
# Initialize Logistic Regression model with quadratic decision boundary
quadratic_model = make_pipeline(PolynomialFeatures(degree=2), LogisticRegression())
quadratic_model.fit(Xtraing, Ytraing)
plt.figure(figsize=(8, 6))
plt.scatter(Xtraing[Ytraing == 'C1'][:, 0], Xtraing[Ytraing == 'C1'][:, 1], edgecolors='k', marker='o', c='orange', label='C1')
plt.scatter(Xtraing[Ytraing == 'C2'][:, 0], Xtraing[Ytraing == 'C2'][:, 1], edgecolors='k', marker='o', c='green', label='C2')
plt.xlabel('X1')
plt.ylabel('X2')
xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
x, y = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
Z = quadratic_model.decision_function(np.c_[x.ravel(), y.ravel()])
Z = Z.reshape(x.shape)
plt.contour(x, y, Z, colors=['red', 'blue'], levels=[0], alpha=0.5, linestyles=['-'])

# Add legend for class labels
plt.legend()
plt.title('Logistic Regression with Quadratic Decision Boundary')
plt.show()
# Compute training accuracy for quadratic model
train_preds_quad = quadratic_model.predict(Xtraing)
train_accuracy_quad = accuracy_score(Ytraing, train_preds_quad)
print(f'Training Accuracy (Quadratic): {train_accuracy_quad}')
# Compute testing accuracy for quadratic model
test_preds_quad = quadratic_model.predict(Xtraing)
test_accuracy_quad = accuracy_score(Ytraing, test_preds_quad)
print(f'Testing Accuracy (Quadratic): {test_accuracy_quad}')


