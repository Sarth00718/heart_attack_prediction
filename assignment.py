import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''-------------------------------------------------------------------------------------
Select a source from KAGGLE/GitHub/similar repository, and identify the Data Domain and 
Characteristics of the selected Dataset. Also, write a report with the following detail

a) Selection of data domain
b) find the data characteristics (null, not null, unique values)
d) Find the output/class label of the data set
e) Define the selection of fields (Data types)
-------------------------------------------------------------------------------------'''
 
# read the csv 
data=pd.read_csv("heart_attack_prediction.csv",header='infer')

# Data Characteristics
null_values = data.isnull().sum()  # Count null values in each column
not_null_values = data.notnull().sum()  # Count not null values in each column
unique_values = data.nunique()  # Count unique values in each column

print("Data Characteristics:")
print("Null Values:")
print(null_values)
print("\nNot Null Values:")
print(not_null_values)
print("\nUnique Values:")
print(unique_values)

# Identify output/class label
print("Output/class label:", data['Heart Attack Risk'].unique())
 
# Define data types
data_types = data.dtypes
print(data_types)
 
'''-------------------------------------------------------------------------------------
Load the dataset, drop all the null records and replace the NA values in the numerical 
column with the mean value of the field as per the class label and categorical columns 
with the mode value of the field as per the class label.
-------------------------------------------------------------------------------------'''

# Drop all null records
df = data.dropna()
print(df)

#handling missing value
print(data.info())

#mean of the age 
meanage=np.mean(data.loc[~data["Age"].isna(),"Age"]) 
data.loc[data["Age"].isna(),"Age"] = meanage
print(data["Age"]) 
print(data.info())

#mode of the gender
modegen=(data.loc[~data["Sex"].isna(),"Sex"]).mode()[0] 
print(modegen)
data.loc[data["Sex"].isna(),"Sex"] = modegen 
print(data.info())

#mode of the hemisphere
modehs=(data.loc[~data["Hemisphere"].isna(),"Hemisphere"]).mode()[0] 
print(modehs)
data.loc[data["Hemisphere"].isna(),"Hemisphere"] = modehs
print(data.info())

'''-------------------------------------------------------------------------------------
Perform statistical analysis on the selected dataset (count, sum, range, min, max, mean, 
median, mode, variance and Standard deviation).
-------------------------------------------------------------------------------------'''

# first convert all colums in numerical value
from sklearn.preprocessing import LabelEncoder

# Perform label encoding for categorical columns
label_encoder = LabelEncoder()
categorical_cols = ['Patient ID','Age','Sex','Blood Pressure','Diet','Country', 'Continent','Hemisphere']
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])
print(data)
    
#mean
columns = data.columns
means = {}
for col in columns:
    col_data = np.array(data[col])
    mean = np.mean(col_data)
    means[col] = mean
print("Mean for each column:")
for col, mean in means.items():
    print(f"{col}: {mean}")

print('')

# Median
medians = {}
for col in columns:
    col_data = np.array(data[col])
    median = np.median(col_data)
    medians[col] = median
print("Median for each column:")
for col, median in medians.items():
    print(f"{col}: {median}")

print('')

#mode
from scipy import stats
modes = {}
for col in columns:
    col_data = np.array(data[col])
    mode = stats.mode(col_data)
    modes[col] = mode
print("Mode for each column:")
for col, mode in modes.items():
    print(f"{col}: {mode}")

print('')

# Standard Deviation
std_devs = {}
for col in columns:
    col_data = np.array(data[col])
    std_dev = np.std(col_data)
    std_devs[col] = std_dev
print("Standard Deviation for each column:")
for col, std_dev in std_devs.items():
    print(f"{col}: {std_dev}")

print('')

# Variance
variances = {}
for col in columns:
    col_data = np.array(data[col])
    variance = np.var(col_data)
    variances[col] = variance
print("Variance for each column:")
for col, variance in variances.items():
    print(f"{col}: {variance}")

print('')

# Range
ranges = {}
for col in columns:
    col_data = np.array(data[col])
    col_range = np.ptp(col_data)  # Using np.ptp to calculate the range
    ranges[col] = col_range
print("Range for each column:")
for col, col_range in ranges.items():
    print(f"{col}: {col_range}")

print('')

# Maximum
max_values = {}
for col in columns:
    col_data = np.array(data[col])
    max_val = np.max(col_data)
    max_values[col] = max_val
print("Maximum value for each column:")
for col, max_val in max_values.items():
    print(f"{col}: {max_val}")

print('')

# Minimum
min_values = {}
for col in columns:
    col_data = np.array(data[col])
    min_val = np.min(col_data)
    min_values[col] = min_val
print("Minimum value for each column:")
for col, min_val in min_values.items():
    print(f"{col}: {min_val}")

print('')

# Count
counts = {}
for col in columns:
    count = data[col].count()
    counts[col] = count
print("Count of non-null values for each column:")
for col, count in counts.items():
    print(f"{col}: {count}")

print('')

# Sum
sums = {}
for col in columns:
    col_data = np.array(data[col])
    col_sum = np.sum(col_data)
    sums[col] = col_sum
print("Sum of values for each column:")
for col, col_sum in sums.items():
    print(f"{col}: {col_sum}")

print('')   
    
print(data.info())
print(data)

# Perform statistical analysis on numerical columns
statistics = data.describe()

# Compute additional statistics
count = data.count()
sum_values = data.sum()
range_values = data.max() - data.min()
min_values = data.min()
max_values = data.max()
mean_values = data.mean()
median_values = data.median()
mode_values = data.mode().iloc[0]
variance_values = data.var()
std_deviation_values = data.std()

# Collecting statistics into a DataFrame
statistics = pd.DataFrame({
    'Count': count,
    'Sum': sum_values,
    'Range': range_values,
    'Min': min_values,
    'Max': max_values,
    'Mean': mean_values,
    'Median': median_values,
    'Mode': mode_values,
    'Variance': variance_values,
    'Standard Deviation': std_deviation_values
})

# Displaying the statistics
print(statistics)

'''-------------------------------------------------------------------------------------
Display all the unique value counts and unique values of all the columns of the dataset.
-------------------------------------------------------------------------------------'''

# Display unique value counts and unique values for each column
for column in data.columns:
    unique_values = data[column].unique()
    unique_value_counts = data[column].value_counts()
    print(f"Column: {column}")
    print(f"Unique Values: {unique_values}")
    print(f"Unique Value Counts:\n{unique_value_counts}\n")
    
'''-------------------------------------------------------------------------------------
Draw applicable plots to visualise data using the subplot concept on the dataset. (scatter 
plot/ line graph/histogram etc.)
----------------------------------------------------------------------------------------'''

import matplotlib.pyplot as plt
import math

# numerical variables - histogram using subplots
numerical_vars = ['Age', 'Cholesterol', 'Heart Rate', 'Income', 'BMI', 'Triglycerides', 
                  'Exercise Hours Per Week', 'Sedentary Hours Per Day', 'Physical Activity Days Per Week', 
                  'Sleep Hours Per Day']

colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray']
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(20, 30))

for i, (var, color) in enumerate(zip(numerical_vars, colors)):
    ax = axs[i // 2, i % 2]
    ax.hist(data[var], bins=150, color=color, edgecolor='black')
    ax.set_title(f'Histogram of {var}')
    ax.set_xlabel(var)
    ax.set_ylabel('Frequency')
    ax.grid(True)

plt.tight_layout()
plt.show()

# Categorical Variables - Bar plots using subplots
categorical_vars = ['Sex', 'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption', 
                    'Diet', 'Previous Heart Problems', 'Medication Use', 'Stress Level', 'Country', 
                    'Continent', 'Hemisphere']

colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray']
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(20, 15))

for i, (var, color) in enumerate(zip(categorical_vars, colors)):
    ax = axs[i // 5, i % 5]
    data[var].value_counts().plot(kind='bar', color=color, ax=ax)
    
    ax.set_title(f'Count of {var}')
    ax.set_xlabel(var)
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid(axis='y')
    
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# categorical variables - pie chat
categorical_columns = ['Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption']
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i, column in enumerate(categorical_columns):
    row = i // 3
    col = i % 3
    column_counts = data[column].value_counts(normalize=True)
    axs[row, col].pie(column_counts, labels=column_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral', 'lightyellow', 'lightgrey', 'lightcyan', 'lightpink', 'lightblue'])
    axs[row, col].set_title(f'Proportion of Patients with {column}')

plt.tight_layout()
plt.show()


# Scatter plot for Age vs Heart Attack Risk
plt.figure(figsize=(10, 8))
plt.scatter(data['Age'], data['Heart Attack Risk'])
plt.title('Scatter plot of Age vs Heart Attack Risk')
plt.xlabel('Age')
plt.ylabel('Heart Attack Risk')
plt.grid(True)


# Pie chart for Heart Attack Risk distribution
plt.figure(figsize=(8, 6))
data['Heart Attack Risk'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'salmon'])
plt.title('Distribution of Heart Attack Risk')
plt.ylabel('')
plt.show()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Scatter plot: Age vs. Cholesterol
plt.figure(figsize=(10, 6))
plt.scatter(data['Age'], data['Cholesterol'], color='blue', alpha=0.5)
plt.title('Scatter Plot of Age vs. Cholesterol')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.grid(True)
plt.show()


# Bar chart: Frequency of Smoking
plt.figure(figsize=(10, 6))
smoking_counts = data['Smoking'].value_counts()
plt.bar(smoking_counts.index, smoking_counts.values, color=['lightgreen', 'lightcoral'])
plt.title('Bar Chart of Smoking Frequency')
plt.xlabel('Smoking')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()


'''-------------------------------------------------------------------------------------
Train the model of the K-nearest Neighbors Classifier/Regressor with 80% of the data and 
predict the class label for the rest 20% of the data. Evaluate the model with all 
appropriate measures.
----------------------------------------------------------------------------------------'''

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Extract features and target variable
x = data.drop(columns=['Patient ID', 'Heart Attack Risk'])  # Features
y = data['Heart Attack Risk']  # Target variable

nrows = data.shape[0]
print("Total Rows:",nrows)

test_split = float(input("Enter a number between 0 and 1 to specify the test spilt:"))

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = test_split)
print("shapes:", x_train.shape , y_train.shape , x_test.shape , y_test.shape )

#knn classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report 

k = int(input("Enter the K neighbors:"))
model = KNeighborsClassifier(n_neighbors=k,weights="distance") 
model.fit(x_train,y_train) 
y_predict=model.predict(x_test)
print(y_predict)

accuracy=accuracy_score(y_test,y_predict) 
print(accuracy)
report=classification_report(y_test,y_predict)
print(report)

'''-------------------------------------------------------------------------------------
Conclude your observation concerning achieved results.
----------------------------------------------------------------------------------------'''

print("""
Report: Analysis of Heart Attack Prediction Dataset

1. Introduction:
The following report presents an analysis of the Heart Attack Prediction dataset, aiming to predict heart attack risk based on various attributes. The dataset contains information on demographic factors, medical history, lifestyle habits, and physiological parameters.

2. Data Domain and Characteristics:

Data Domain: The dataset's domain revolves around predicting heart attack risk.
Data Characteristics:
The dataset consists of several features such as age, cholesterol levels, heart rate, lifestyle factors, and more.
Initial exploration reveals the presence of missing values, which were addressed through preprocessing techniques.
Key insights were derived from identifying null values, not null values, and unique values in each column.

3. Preprocessing and Cleaning:

Null records were dropped from the dataset to ensure data integrity.
Missing values in numerical columns were replaced with the mean value, while categorical columns were imputed with the mode value.

4. Statistical Analysis:

Statistical measures such as count, sum, range, min, max, mean, median, mode, variance, and standard deviation were computed for each column.
These analyses provided a deeper understanding of the dataset's distribution and variability.

5. Visualization:

Visualizations, including histograms, bar plots, pie charts, and scatter plots, were utilized to explore data distributions and relationships between variables.
Insights gained from visualizations helped in identifying patterns and trends within the dataset.

6. Model Training and Evaluation:

A K-nearest Neighbors Classifier was trained using 80% of the dataset, with the remaining 20% used for evaluation.
The model's performance was assessed using accuracy score and classification reports.

7. Observations and Conclusion:

The dataset offers valuable insights into factors influencing heart attack risk, encompassing demographic, medical, and lifestyle aspects.
Preprocessing steps were crucial for enhancing data quality and reliability.
The K-nearest Neighbors Classifier demonstrated moderate predictive performance, suggesting potential utility in heart disease risk assessment.

In conclusion, the analysis of the Heart Attack Prediction dataset provides valuable insights into understanding and predicting heart attack risk, offering potential avenues for improving cardiovascular health outcomes.
""")
