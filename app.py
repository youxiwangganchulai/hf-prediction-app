import streamlit as st
import pandas as pd
import joblib

# ======================
# 加载模型
# ======================

model = joblib.load("heart_model.pkl")

# ======================
# 页面标题
# ======================

st.title("HF Prediction System")

st.write("COPD combined heart failure risk prediction")

# ======================
# 输入变量
# ======================

RF_1 = st.number_input("RF_1", 0.0, 100.0, 1.0)

age = st.number_input("age", 0.0, 120.0, 60.0)

SIICI = st.number_input("SIICI", 0.0, 100000.0, 500.0)

TBIL = st.number_input("TBIL", 0.0, 500.0, 10.0)

PH = st.number_input("PH", 6.0, 8.0, 7.4)

CREA = st.number_input("CREA", 0.0, 2000.0, 80.0)

# ======================
# 构建输入数据
# ======================

input_df = pd.DataFrame({
    "RF_1":[RF_1],
    "age":[age],
    "SIICI":[SIICI],
    "TBIL":[TBIL],
    "PH":[PH],
    "CREA":[CREA]
})

# ======================
# 预测
# ======================

if st.button("Predict"):

    pred_prob = model.predict_proba(input_df)[0,1]

    st.subheader("Prediction Probability")

    st.write(f"HF Risk Probability: {pred_prob:.3f}")

    if pred_prob > 0.5:
        st.error("High Risk")
    else:
        st.success("Low Risk")
