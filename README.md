
# Exploratory Data Analysis (EDA) - Linear Regression Model  

## Project Details  
**Name:** Yogesh Kumar  
**Company:** CODTECH IT SOLUTIONS  
**Employee ID:** CT12WDS88
**Domain:** Data Analytics  
**Duration:** 5th December 2024 to 5th March 2025  
**Mentor:** Neela Santhosh Kumar  

## Project Overview  

This project focuses on **Exploratory Data Analysis (EDA)** and implements a linear regression model to analyze and visualize a synthetic dataset. The objective is to understand the relationship between independent and dependent variables by applying regression techniques, evaluating model performance, and visualizing the results.  

### Key Highlights  
- Dataset generation using NumPy.  
- Linear regression model implementation with scikit-learn.  
- Performance evaluation using Mean Squared Error (MSE) and R-squared (R²).  
- Data visualization with Matplotlib.  



## Prerequisites  

Ensure you have the following Python libraries installed:  
```bash
pip install numpy matplotlib scikit-learn
```  

---

## Code Breakdown  

### 1. Dataset Generation  
A synthetic dataset is generated based on the linear equation:  
\[
y = 4 + 3X + \text{noise}
\]  
Noise is introduced to make the dataset realistic.  

```python
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
```  

### 2. Data Splitting  
The dataset is split into **training (80%)** and **testing (20%)** sets.  

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```  

### 3. Model Training  
A linear regression model is trained on the training dataset.  

```python
model = LinearRegression()
model.fit(X_train, y_train)
```  

### 4. Predictions  
The model predicts values for the test dataset.  

```python
y_pred = model.predict(X_test)
```  

### 5. Model Evaluation  
Model performance is evaluated using:  
- **Mean Squared Error (MSE)**  
- **R-squared (R²)**  

```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```  

### 6. Visualization  
- **Regression Line**: Plots the actual values and the regression line.  
- **Actual vs. Predicted**: Scatter plot comparing actual and predicted values.  

```python
plt.scatter(X, y, color='blue', label='Actual values')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

plt.scatter(y_test, y_pred, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
``` 

---

## Sample Outputs  

- **Mean Squared Error**: `0.58` (example value)  
- **R-squared**: `0.92` (example value)  

## Visualization   
![image](https://github.com/user-attachments/assets/456d6dd4-01ad-4ed0-a0e7-c5976b99a81a)
![image](https://github.com/user-attachments/assets/09761a3a-4c65-424e-909b-1807597068c7)







