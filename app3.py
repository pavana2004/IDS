import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")  

@st.cache_resource
def load_label_encoder():
    return joblib.load("label_encoder.pkl")  
clf = load_model()
label_encoder = load_label_encoder()

st.title("🔍 Network Traffic Analyzer & Predictor")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  
    st.write("### 🔢 Data Preview", df.head())

    features = df.copy()
    if "Label" in features.columns:
        features = features.drop(columns=["Label"])

    features = features.replace([float("inf"), float("-inf")], 0)
    features = features.fillna(0)

    try:
        preds = clf.predict(features)
        df["Prediction"] = preds
        label_map = {1: "Benign", 0: "Attack"}
        df["Prediction_Label"] = df["Prediction"].map(label_map)

        st.write("### 📈 Predictions", df[["Prediction", "Prediction_Label"]].head())
        
        st.write("### 📊 Prediction Distribution")
        pred_counts = df["Prediction_Label"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=pred_counts.index, y=pred_counts.values, palette="coolwarm", ax=ax)
        ax.set_xlabel("Prediction Label")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Predicted Labels")
        st.pyplot(fig)

        if "Label" in df.columns:
            try:
                
                df["Label"] = df["Label"].astype(str).str.strip().str.upper()
                df["True_Label"] = df["Label"].apply(lambda x: 1 if x == "BENIGN" else 0)


                st.write("### 📊 Classification Report")
                report = classification_report(df["True_Label"], preds, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                st.write("### 🧮 Confusion Matrix")
                cm = confusion_matrix(df["True_Label"], preds)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                

            except Exception as e:
                st.warning(f"⚠️ Could not apply label encoder: {e}")

        st.write("### 🔬 Per Row Analysis")
        for i, row in df.iterrows():
            st.markdown(f"**Row {i+1}:**")
            st.json({
                "Prediction": int(row["Prediction"]),
                "Status": "✅ Benign" if row["Prediction"] == 0 else "⚠️ Attack",
                "Flow Duration": row.get("Flow Duration", "N/A"),
                "Destination Port": row.get("Destination Port", "N/A"),
                "Flow Bytes/s": row.get("Flow Bytes/s", "N/A"),
                "Packet Length Mean": row.get("Packet Length Mean", "N/A"),
                "Flow IAT Mean": row.get("Flow IAT Mean", "N/A")
            })
            if i >= 9:
                st.info("Showing only first 10 rows for brevity.")
                break

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", data=csv, file_name='predictions.csv', mime='text/csv')

    except Exception as e:
        st.error(f"Prediction failed: {e}")
