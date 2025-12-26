import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def main():
    st.title("🌧️ Rainfall Prediction Using Linear Regression")
    st.write("Upload a CSV dataset and predict rainfall using Linear Regression.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a CSV file to begin.")
        return

    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write(data.head())

    # Select target column
    target = st.selectbox("Select Target Column (Rainfall)", data.columns)

    # Feature selection
    features = st.multiselect(
        "Select Feature Columns",
        [col for col in data.columns if col != target]
    )

    if not features:
        st.warning("Select at least one feature to train the model.")
        return

    X = data[features]
    y = data[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Performance")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # Plot Actual vs Predicted
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Rainfall")
    ax.set_ylabel("Predicted Rainfall")
    ax.set_title("Actual vs Predicted Rainfall")
    st.pyplot(fig)

    # Prediction section
    st.subheader("🔮 Predict Rainfall")

    input_values = {}
    cols = st.columns(len(features)) if len(features) <= 6 else None
    for i, feature in enumerate(features):
        if cols:
            with cols[i]:
                input_values[feature] = st.number_input(f"{feature}", value=0.0)
        else:
            input_values[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button("Predict"):
        row = [input_values[f] for f in features]
        prediction = model.predict([row])
        st.success(f"🌧️ Predicted Rainfall: {prediction[0]:.2f} mm")


if __name__ == "__main__":
    main()
