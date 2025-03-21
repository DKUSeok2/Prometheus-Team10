from sklearn.model_selection import train_test_split
import pandas as pd

# ✅ 2️⃣ Parquet 데이터 로드
df = pd.read_parquet("Save/summary_com.parquet")

# ✅ 3️⃣ Train/Validation/Test 데이터 분할 (80/10/10)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# ✅ 4️⃣ 분할된 데이터 저장
train_df.to_parquet("Save/train.parquet", index=False)
valid_df.to_parquet("Save/valid.parquet", index=False)
test_df.to_parquet("Save/test.parquet", index=False)
