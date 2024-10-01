import streamlit as st
import pickle
import numpy as np


# Load the model and necessary encoders
def load_model():
    with open('C:/Users/user/PycharmProjects/pythonProject1/pythonProject1/ml2/saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

regressor_loaded = data["model"]
le_company = data["le_company"]
le_storagetype = data["le_storagetype"]


def show_predict_page():
    st.title("Laptop Price Prediction in Euros")

    st.write("""### We need some laptop properties to predict a laptop price""")

    # Main page widgets
    Companies = ["Dell", "Lenovo", "HP", "Asus", "Acer", "MSI", "Other", "Toshiba", "Apple"]
    Storagetypes = ["SSD", "Flash Storage", "HDD", "Hybrid"]

    company = st.selectbox("Company", Companies)
    storagetype = st.selectbox("Storagetype", Storagetypes)
    Ram = st.slider("Ram", 0, 32, 2)
    storage = st.slider("Storage", 0, 2048, 16)

    ok = st.button("Calculate Price")

    if ok:
        # Prepare the input array for prediction
        x = np.array([[company, Ram, storage, storagetype]])

        # Transform categorical inputs
        x[:, 0] = le_company.transform(x[:, 0])
        x[:, 3] = le_storagetype.transform(x[:, 3])

        # Predict price using the loaded model
        Price = regressor_loaded.predict(x)
        st.subheader(f"The estimated Price is: €{Price[0]:.2f}")







