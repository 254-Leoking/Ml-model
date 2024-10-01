import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile


# Define the path to your zip file
zip_file_path = r'C:\Users\user\Downloads\archive (2).zip'

# Check if the zip file exists
if not os.path.exists(zip_file_path):
    print("The specified zip file does not exist.")
else:
    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        # List the contents of the zip file
        print(z.namelist())  # This will show all the files inside the zip

        # Extract the desired CSV file
        with z.open('laptop_prices.csv') as csvfile:
            df = pd.read_csv(csvfile)

    # Display the DataFrame
    print(df.head())


# Checking null values.
print(df.info())
# Checking every row that is duplicated
print(df.duplicated())
# Keeping the needed rows for the model.
df = df[["Company", "Ram", "PrimaryStorage", "PrimaryStorageType", "Price_euros"]]
df = df.rename({"PrimaryStorage": "Storage"}, axis=1)
df = df.rename({"PrimaryStorageType": "StorageType"}, axis=1)
df.head()

print(df.head())

print(df.info())
# Drop row with no values.
df = df.dropna()
df.isnull().sum()

df['Company'].value_counts()
print(df['Company'].value_counts())


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


company_map = shorten_categories(df.Company.value_counts(), 20)
df['Company'] = df['Company'].map(company_map)
print(df.Company.value_counts())
# Plot prices against company to get its range.
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
df.boxplot('Price_euros', 'Company', ax=ax)
plt.suptitle('Prices in Euros  v Company')
plt.title('')
plt.ylabel('Prices')
plt.xticks(rotation=90)
plt.show()

df = df[df["Price_euros"] <= 2000]
df = df[df["Price_euros"] > 0]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
df.boxplot('Price_euros', 'Company', ax=ax)
plt.suptitle('Prices in euros v Company')
plt.title('')
plt.ylabel('Laptops Prices')
plt.xticks(rotation=90)
plt.show()

print(df["Ram"].unique())

print(df["Storage"].unique())

print(df["StorageType"].unique())


def clean_storagetype(x):
    if "SSD" in x:
        return "SSD"
    if "Flash Storage" in x:
        return "Flash Storage"
    if "HDD" in x:
        return "HDD"
    if "Hybrid" in x:
        return "Hybrid"


df["StorageType"] = df["StorageType"].apply(clean_storagetype)

from sklearn.preprocessing import LabelEncoder
le_storagetype = LabelEncoder()
df["StorageType"] = le_storagetype.fit_transform(df["StorageType"])
print(df["StorageType"].unique())

le_company = LabelEncoder()
df["Company"] = le_company.fit_transform(df["Company"])
print(df["Company"].unique())

x = df.drop("Price_euros", axis=1)
y = df["Price_euros"]

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x, y.values)

LinearRegression()

y_pred = linear_reg.predict(x)

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
error = np.sqrt(mean_squared_error(y, y_pred))

print(error)

# The error is pretty high we try a different model.
from  sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(x, y.values)

DecisionTreeRegressor(random_state=0)
y_pred = dec_tree_reg.predict(x)

error = np.sqrt(mean_squared_error(y, y_pred))
print(error)
regressor = dec_tree_reg
regressor.fit(x, y.values)
y_pred = regressor.predict(x)
error = np.sqrt(mean_squared_error(y, y_pred))

print(x)


x = np.array([["Apple", 16, 128, "SSD"]])

print(x)

x[:, 0] = le_company.transform(x[:, 0])
x[:, 3] = le_storagetype.transform(x[:, 3])

print(x)

y_pred = regressor.predict(x)
print(y_pred)

import pickle

data = {"model": regressor, "le_company": le_company, "le_storagetype": le_storagetype}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)

with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data["model"]
le_company = data["le_company"]
le_storagetype = data["le_storagetype"]

y_pred = regressor_loaded.predict(x)
print(y_pred)

























