# Giới hạn đã biết

- Nhãn ML do quy tắc sinh, không phải phản hồi người dùng; chỉ số dự đoán không đo chất lượng gu thời trang thực.
- Không có rating/feedback outfit đủ để đánh giá relevance thực tế hoặc huấn luyện Collaborative Filtering.
- Các metric ranking hiện có là proxy dùng nhãn quy tắc; coverage/diversity đo trên 20 bối cảnh tổng hợp.
- Mô hình không dùng giới tính, lịch, vị trí, mưa hay gió làm đặc trưng RF–KNN; một số trường có thể tham gia luật/giải thích riêng.
- Ba trong 20 bối cảnh tổng hợp trả ít hơn số outfit yêu cầu; hệ thống trả cảnh báo thay vì đảm bảo đủ số lượng.
- Không thể xác nhận UI bằng ảnh chụp đa viewport hoặc đo tải thực trong lần chạy này.
- Nhóm kiểm thử FastAPI TestClient đầy đủ bị chặn do sandbox không cho tạo socketpair cục bộ; không ghi nhận nhóm này là đạt.
