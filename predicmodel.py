# Import thư viện
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
# Import thêm các thư viện cần thiết
# from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_categorical_dtype

# Đọc dữ liệu
data = pd.read_csv('C:/Project/Dự đoán bệnh suy tim/heart.csv')

# Tiền xử lý dữ liệu
data['ExerciseAngina'] = data['ExerciseAngina'].map({'N': 0, 'Y': 1})
data['HeartDisease'] = data['HeartDisease'].astype(int)

# Tách features và target
target = 'HeartDisease'
x = data.drop(target, axis=1)
y = data[target]

# Chia dữ liệu thành tập huấn luyện và kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
preprocessor = ColumnTransformer(transformers=[
    ("one_hot_feature", OneHotEncoder(), ["ChestPainType", "RestingECG", "ST_Slope"]),
    ("ordinal_feature", OrdinalEncoder(), ["Sex"]),
    ("standard_feature", StandardScaler(), ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]),
])

# Xây dựng pipeline
reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

# Huấn luyện mô hình
reg.fit(x_train, y_train)

# Dự đoán
y_predict = reg.predict(x_test)

# Tính các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average="binary")
recall = recall_score(y_test, y_predict, average="binary")
f1 = f1_score(y_test, y_predict, average="binary")

# Streamlit UI
st.title("Heart Disease Prediction App")
st.markdown("""
### 🫀 Dự đoán bệnh suy tim
Ứng dụng sử dụng mô hình Random Forest để phân loại nguy cơ mắc bệnh suy tim dựa trên thông tin sức khỏe và các yếu tố liên quan.
""")

# Khám phá dữ liệu
st.subheader("🔍 Dữ liệu ban đầu")
st.write(data.head())

# Tổng quan về dữ liệu
st.subheader("📋 Tổng quan về dữ liệu")
num_observations = data.shape[0]
num_variables = data.shape[1]
missing_cells = data.isnull().sum().sum()
missing_cells_percent = (missing_cells / (num_observations * num_variables)) * 100
duplicate_rows = data.duplicated().sum()
duplicate_rows_percent = (duplicate_rows / num_observations) * 100
total_size = data.memory_usage(deep=True).sum() / 1024  # Kích thước (KB)
avg_record_size = total_size * 1024 / num_observations  # Kích thước trung bình (B)

# Hiển thị tổng quan
st.write(f"**Số lượng biến:** {num_variables}")
st.write(f"**Số lượng quan sát:** {num_observations}")
st.write(f"**Số lượng ô trống:** {missing_cells} ({missing_cells_percent:.1f}%)")
st.write(f"**Số hàng trùng lặp:** {duplicate_rows} ({duplicate_rows_percent:.1f}%)")
st.write(f"**Tổng dung lượng trong bộ nhớ:** {total_size:.1f} KiB")
st.write(f"**Kích thước trung bình của mỗi bản ghi:** {avg_record_size:.1f} B")

# Thêm biểu đồ tương tác
st.sidebar.header("🔄 Tương tác giữa các biến")
x_axis = st.sidebar.selectbox("Chọn biến trục X:", options=data.columns, index=0)
y_axis = st.sidebar.selectbox("Chọn biến trục Y:", options=data.columns, index=88)

# Vẽ biểu đồ tương tác
st.subheader("📊 Biểu đồ tương tác")
fig = px.scatter(
    data,
    x=x_axis,
    y=y_axis,
    color="HeartDisease",
    title=f"Tương tác giữa {x_axis} và {y_axis}",
    labels={x_axis: x_axis, y_axis: y_axis},
    hover_data=data.columns
)
st.plotly_chart(fig)
# Đánh giá mô hình
st.subheader("📊 Đánh giá mô hình")
st.write(f"**Độ chính xác (Accuracy):** {accuracy:.2f}")   
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**F1 Score:** {f1:.2f}")

# Nhập dữ liệu đầu vào từ người dùng
st.sidebar.header("🏥 Nhập thông tin bệnh nhân")
age = st.sidebar.number_input("Tuổi", min_value=20, max_value=100, value=50)
sex = st.sidebar.selectbox("Giới tính", ["M", "F"])
resting_bp = st.sidebar.number_input("Huyết áp khi nghỉ ngơi (mmHg)", min_value=80, max_value=200, value=120)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
fasting_bs = st.sidebar.selectbox("Lượng đường máu đói > 120 mg/dL", ["Không", "Có"])
max_hr = st.sidebar.number_input("Nhịp tim tối đa", min_value=60, max_value=220, value=150)
exercise_angina = st.sidebar.selectbox("Đau thắt ngực khi tập thể dục", ["Không", "Có"])
oldpeak = st.sidebar.number_input("ST trầm cảm", min_value=-5.0, max_value=10.0, value=1.0, step=0.1)
chest_pain_type = st.sidebar.selectbox("Loại đau ngực", data["ChestPainType"].unique())
resting_ecg = st.sidebar.selectbox("Kết quả điện tâm đồ", data["RestingECG"].unique())
st_slope = st.sidebar.selectbox("Độ dốc đoạn ST", data["ST_Slope"].unique())

# Tiền xử lý dữ liệu đầu vào
input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "RestingBP": [resting_bp],
    "Cholesterol": [cholesterol],
    "FastingBS": [1 if fasting_bs == "Có" else 0],
    "MaxHR": [max_hr],
    "ExerciseAngina": [1 if exercise_angina == "Có" else 0],
    "Oldpeak": [oldpeak],
    "ChestPainType": [chest_pain_type],
    "RestingECG": [resting_ecg],
    "ST_Slope": [st_slope]
})

# Dự đoán và hiển thị kết quả
predicted_heart_disease = reg.predict(input_data)[0]
result = "Nguy cơ cao mắc bệnh tim" if predicted_heart_disease == 1 else "Ít nguy cơ mắc bệnh tim"

st.subheader("📋 Kết quả dự đoán")
st.write(f"**Kết quả:** {result}")


