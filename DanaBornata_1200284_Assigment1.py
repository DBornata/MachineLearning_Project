import matplotlib.pyplot as plt
import numpy as np
#1- Read the dataset and examine how many features and examples does it have? (Hint: you can use Pandas to load the dataset into a dataframe)
# Load the dataset into a Pandas DataFrame
FILE= "C:/Users/Lenovo/Desktop/Dana/cars.csv"  
data = pd.read_csv(FILE)
print("\n\t\t1)\n")
# Extract 'horsepower' and 'mpg' columns as numpy arrays
X = data['horsepower'].values
y = data['mpg'].values

# Feature normalization
X_normalized = (X - np.mean(X)) / np.std(X)

# Add a column of ones for the intercept term
X_with_intercept = np.vstack([np.ones_like(X_normalized), X_normalized]).T

# Set hyperparameters
learning_rate = 0.001
num_iterations = 80000
theta = np.zeros(X_with_intercept.shape[1])

for iteration in range(num_iterations):
    predictions = X_with_intercept @ theta
    error = predictions - y
    gradient = X_with_intercept.T @ error
    theta -= learning_rate * gradient / len(y)



print(f"Learned coefficients using gradient descent: {theta}")

# Plot the scatter plot
sns.scatterplot(x='horsepower', y='mpg', data=data, color='black', alpha=0.7)

plt.plot(X, X_with_intercept @ theta, color='red', label='Linear Regression Line (Gradient Descent)')

# Set plot labels and title
plt.xlabel('Horsepower')
plt.ylabel('mpg')
plt.title('Linear Regression (Gradient Descent): Horsepower vs. mpg')

# Display the legend
plt.legend()

# Show the plot
plt.show()