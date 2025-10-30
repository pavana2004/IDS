import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import joblib
import os

# === CONFIG ===
DATA_DIR = "dataset"  
MODEL_OUTPUT = "rf_model.pkl"
ENCODER_OUTPUT = "label_encoder.pkl"
N_TREES = 100

print("🔄 Loading dataset...")

all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
df_list = []

for f in all_files:
    try:
        df = pd.read_csv(f, low_memory=False)
        df_list.append(df)
    except Exception as e:
        print(f"❌ Error loading {f}: {e}")

combined_df = pd.concat(df_list, ignore_index=True)
print(f"Loaded {len(combined_df)} rows.")

print(" Preprocessing...")

combined_df = combined_df.drop(columns=["Flow ID", "Source IP", "Destination IP", "Timestamp"], errors='ignore')

combined_df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
combined_df.dropna(inplace=True)
combined_df.columns = combined_df.columns.str.strip()

# Label encoding: 1 = Benign, 0 = Attack
combined_df["Label"] = combined_df["Label"].apply(lambda x: "Attack" if str(x).upper() == "ATTACK" else "Benign")
le = LabelEncoder()
combined_df["Label"] = le.fit_transform(combined_df["Label"])

X = combined_df.drop("Label", axis=1)
y = combined_df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"🌲 Training Random Forest with {N_TREES} trees...")

clf = RandomForestClassifier(
    n_estimators=1,
    warm_start=True,
    random_state=42,
    max_depth=None,
    n_jobs=-1,
    class_weight='balanced'
)

for i in tqdm(range(1, N_TREES + 1), desc="Training Trees"):
    clf.set_params(n_estimators=i)
    clf.fit(X_train, y_train)

print("📊 Evaluation:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(clf, MODEL_OUTPUT)
joblib.dump(le, ENCODER_OUTPUT)

print(f"✅ Model saved to: {MODEL_OUTPUT}")
print(f"✅ Label encoder saved to: {ENCODER_OUTPUT}")
