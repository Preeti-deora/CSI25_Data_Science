import streamlit as st
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
import numpy as np

df = sns.load_dataset("mpg").dropna()
model = joblib.load("Assignment_week7/mpg_model.pkl")

df['origin'] = df['origin'].astype('category').cat.codes
df['name'] = df['name'].astype('category').cat.codes

X = df.drop("mpg", axis=1)
y = df["mpg"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.set_page_config(page_title="MPG Predictor", layout="centered", page_icon="⛽")

st.title("⛽ MPG Prediction App")
st.caption("Predict a car's fuel efficiency (MPG) and understand the model's decision process.")
st.divider()

st.sidebar.header("📥 Input Parameters")
def user_input():
    cylinders = st.sidebar.slider("Cylinders", int(df.cylinders.min()), int(df.cylinders.max()), 4)
    displacement = st.sidebar.slider("Displacement", float(df.displacement.min()), float(df.displacement.max()), 150.0)
    horsepower = st.sidebar.slider("Horsepower", float(df.horsepower.min()), float(df.horsepower.max()), 100.0)
    weight = st.sidebar.slider("Weight", float(df.weight.min()), float(df.weight.max()), 2800.0)
    acceleration = st.sidebar.slider("Acceleration", float(df.acceleration.min()), float(df.acceleration.max()), 15.0)
    model_year = st.sidebar.slider("Model Year", int(df.model_year.min()), int(df.model_year.max()), 76)
    origin = st.sidebar.selectbox("Origin", ['usa', 'europe', 'japan'])
    name = st.sidebar.selectbox("Car Name", df['name'].astype('category').cat.categories)

    origin_encoded = {'usa': 2, 'europe': 0, 'japan': 1}[origin]
    name_encoded = df[df['name'] == name]['name'].astype('category').cat.codes.iloc[0]

    input_dict = {
        'cylinders': cylinders,
        'displacement': displacement,
        'horsepower': horsepower,
        'weight': weight,
        'acceleration': acceleration,
        'model_year': model_year,
        'origin': origin_encoded,
        'name': name_encoded
    }
    return pd.DataFrame(input_dict, index=[0])

input_df = user_input()

st.subheader("🔧 Entered Car Specifications")
st.dataframe(input_df, use_container_width=True)
st.divider()

prediction = model.predict(input_df)[0]
st.subheader("🎯 Predicted MPG")
st.success(f"Estimated MPG: **{prediction:.2f}**")

if prediction >= 30:
    st.info("✅ Excellent fuel efficiency!")
elif prediction >= 20:
    st.warning("⚠️ Moderate fuel efficiency.")
else:
    st.error("❌ Poor fuel efficiency.")

st.divider()

st.subheader("📊 Feature Importance")
importances = model.feature_importances_
feat_imp_df = pd.Series(importances, index=X.columns).sort_values()

fig1, ax1 = plt.subplots(figsize=(8, 5))
feat_imp_df.plot(kind='barh', color=plt.cm.viridis_r(np.linspace(0, 1, len(feat_imp_df))), ax=ax1)
ax1.set_title("Which Features Influence MPG Most", fontsize=14)
ax1.set_xlabel("Importance")
ax1.set_ylabel("Feature")
st.pyplot(fig1)
st.divider()

st.subheader("📉 Actual vs Predicted MPG (Test Data)")
y_pred = model.predict(X_test)

fig2, ax2 = plt.subplots(figsize=(6, 5))
sc = ax2.scatter(y_test, y_pred, c=y_test, cmap='coolwarm', edgecolors='k')
ax2.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='gray')
ax2.set_xlabel("Actual MPG")
ax2.set_ylabel("Predicted MPG")
ax2.set_title("Model Accuracy Comparison")
fig2.colorbar(sc, label='True MPG')
st.pyplot(fig2)
st.divider()

selected_feature = st.selectbox("🔍 Explore Feature Impact (Partial Dependence)", X.columns)
st.write(f"Showing how `{selected_feature}` influences predicted MPG on average.")

fig3, ax3 = plt.subplots(figsize=(7, 4))
PartialDependenceDisplay.from_estimator(model, X_test, [selected_feature], ax=ax3)
st.pyplot(fig3)
