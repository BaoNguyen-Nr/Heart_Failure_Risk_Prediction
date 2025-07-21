🫀 Ứng dụng Dự đoán Bệnh Suy Tim
Một ứng dụng Streamlit sử dụng mô hình Random Forest để phân loại và dự đoán nguy cơ mắc bệnh tim mạch dựa trên dữ liệu y tế cá nhân.
🚀 Tính năng nổi bật
Nhập thông tin bệnh nhân từ giao diện người dùng.
Hiển thị biểu đồ tương tác (sử dụng Plotly).
Dự đoán nguy cơ mắc bệnh tim theo thời gian thực.
Phân tích và đánh giá mô hình (accuracy, precision, recall, F1).
Tổng quan dữ liệu: thiếu dữ liệu, trùng lặp, kích thước bộ nhớ,...
🧠 Mô hình học máy
Thuật toán: Random Forest Classifier
Tiền xử lý:
One-Hot Encoding cho các cột phân loại (ChestPainType, RestingECG, ST_Slope)
Ordinal Encoding cho giới tính (Sex)
Standard Scaling cho các đặc trưng số
Chia dữ liệu: 80% huấn luyện, 20% kiểm tra
Độ chính xác (Accuracy): 0.89
🏃‍♂️ Cách chạy ứng dụng
1. Clone dự án
bash
Copy
Edit
git clone https://github.com/BaoNguyen-Nr/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
2. Cài đặt thư viện
bash
Copy
Edit
pip install -r requirements.txt
Các thư viện cần thiết:
pandas, scikit-learn, streamlit, plotly

3. Chạy ứng dụng Streamlit
bash
Copy
Edit
streamlit run app.py
📊 Biểu đồ tương tác
Ứng dụng cung cấp biểu đồ phân tích tương tác giữa các biến đầu vào và nguy cơ bệnh tim, ví dụ:
Tuổi vs Nhịp tim tối đa
Cholesterol vs ST trầm cảm
Tình trạng đau thắt ngực khi tập thể dục
📁 Cấu trúc thư mục
css
Copy
Edit
├── app.py                  # Mã nguồn chính của ứng dụng
├── heart.csv              # Dữ liệu y tế đầu vào
├── README.md              # File hướng dẫn (bạn đang đọc)
├── requirements.txt       # Thư viện cần cài đặt
📝 Dữ liệu đầu vào
Bộ dữ liệu được sử dụng có các trường sau:
Age
Sex
RestingBP
Cholesterol
FastingBS
MaxHR
ExerciseAngina
Oldpeak
ChestPainType
RestingECG
ST_Slope
HeartDisease (biến mục tiêu: 0 - không, 1 - có)
📌 Giao diện mẫu
<img src="https://user-images.githubusercontent.com/your_demo_image.png" alt="Demo UI" width="600"/>
✅ Đánh giá mô hình
Metric	Value
Accuracy	0.89
Precision	~0.88
Recall	~0.90
F1 Score	~0.89
👨‍⚕️ Lưu ý
Ứng dụng này chỉ phục vụ mục đích giáo dục và minh họa kỹ thuật. Không được sử dụng để chẩn đoán y tế thực tế.
