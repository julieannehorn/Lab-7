import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ---- Step 1: Load and Preprocess the Dataset ----
@st.cache_data
def load_data():
    df = pd.read_csv("AmesHousing.csv")  # Ensure this file is in the same folder
    df = df.drop(columns=['PID'], errors='ignore')  # Remove unnecessary columns
    df = df.fillna("None")  # Handle missing values
    df = pd.get_dummies(df, drop_first=True)  # Convert categorical variables
    return df

df = load_data()

# ---- Step 2: Split Data into Training and Testing Sets ----
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Step 3: Train the Model ----
model = LinearRegression()
model.fit(X_train, y_train)

# ---- Step 4: Save the Model ----
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# ---- Step 5: Load Model for Streamlit App ----
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# ---- Step 6: Build Streamlit Web App ----
st.title("🏡 Ames Housing Price Predictor")

st.write("Move the sliders below to adjust house parameters and get a price prediction:")

# ---- User Input Fields (Using Sliders) ----
lot_area = st.slider("Lot Area (sq ft)", min_value=500, max_value=50000, value=7500, step=100)
year_built = st.slider("Year Built", min_value=1800, max_value=2025, value=2000, step=1)

# ---- Make Prediction ----
if st.button("Predict Price"):
    input_data = np.array([[lot_area, year_built]])  # Prepare input

    try:
        prediction = model.predict(input_data)[0]  # Predict
        st.success(f"🏠 Predicted House Price: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

st.write("Made with ❤️ using Streamlit")
