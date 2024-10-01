import streamlit as st
import pandas as pd
import os
import zipfile
import matplotlib.pyplot as plt


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


def clean_storagetype(x):
    if "SSD" in x:
        return "SSD"
    if "Flash Storage" in x:
        return "Flash Storage"
    if "HDD" in x:
        return "HDD"
    if "Hybrid" in x:
        return "Hybrid"


@st.cache
def load_data():
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
    df = df[["Company", "Ram", "PrimaryStorage", "PrimaryStorageType", "Price_euros"]]
    df = df.rename({"PrimaryStorage": "Storage"}, axis=1)
    df = df.rename({"PrimaryStorageType": "StorageType"}, axis=1)
    company_map = shorten_categories(df.Company.value_counts(), 20)
    df['Company'] = df['Company'].map(company_map)
    df = df[df["Price_euros"] <= 2000]
    df = df[df["Price_euros"] > 0]

    df["StorageType"] = df["StorageType"].apply(clean_storagetype)
    return df


df = load_data()


def show_explore_page():
    st.title('Explore Laptops Prices')

    st.write(
        """
        ### Laptop prices
        """
    )

    data = df['Company'].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')

    st.write("""### Number of Laptops from different Companies""")

    st.pyplot(fig1)

    st.write(
        """
        ### Mean Price of Laptops based on Companies
        """
    )

    data = df.groupby(["Company"])["Price_euros"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
        """
        Mean Price of Laptops based on Storage Type
        """
    )

    data = df.groupby(["StorageType"])["Price_euros"].mean().sort_values(ascending=True)
    st.line_chart(data)




