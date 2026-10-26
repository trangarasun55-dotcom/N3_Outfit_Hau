# Kiến trúc DSS khuyến nghị

Hệ thống sử dụng Content-Based Filtering (CBF), mô hình dự đoán lai RF–KNN và luật DSS. Dữ liệu tương tác rating/feedback thực không có trong snapshot nên Collaborative Filtering chưa phù hợp.

1. **Hồ sơ yêu cầu:** bối cảnh, nhiệt độ, phong cách, màu và mức trang trọng do người dùng nhập; đây là hồ sơ tức thời, không phải hồ sơ học từ lịch sử.
2. **Lọc nội dung:** cosine similarity giữa các thuộc tính sản phẩm và yêu cầu; lọc giới tính, nhiệt độ, loại sản phẩm và ưu tiên phong cách tạo ứng viên theo slot.
3. **Chấm điểm:** mô hình lai dùng 95% RF và 5% KNN; điểm phù hợp quy tắc và độ tin cậy được giữ riêng.
4. **Luật ghép DSS:** ghép các slot trang phục, kiểm tra điều kiện thời tiết/loại sản phẩm, tránh lặp sản phẩm chính và giải thích kết quả. MMR tăng đa dạng giữa các bộ.
5. **Cold start:** CBF dùng thuộc tính danh mục và đầu vào tường minh nên có thể xử lý sản phẩm mới có đủ thuộc tính. Không có cá nhân hóa lịch sử.

Phân biệt: ML dự đoán nhãn điểm quy tắc; CBF tìm sản phẩm phù hợp nội dung; DSS áp ràng buộc và xếp bộ trang phục. Xem `api/recommender.py`, `api/rules.py`, `api/config.py` để đối chiếu triển khai.
