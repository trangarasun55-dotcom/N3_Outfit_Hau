# Luật DSS và cách xử lý

Các luật được thực thi trong recommender/rules và cấu hình API; điểm ML không thay thế luật ghép.

- Thuộc tính yêu cầu đầu vào gồm bối cảnh, nhiệt độ, giới tính, hoạt động/lịch, phong cách, màu và mức trang trọng tùy endpoint.
- Ứng viên được giới hạn theo nhóm sản phẩm, giới tính và điều kiện nhiệt độ; hồ sơ tường minh dùng cosine similarity cho phong cách, màu và mức trang trọng.
- Điểm RF–KNN xếp hạng khả năng phù hợp với nhãn điểm quy tắc. Luật ghép kiểm soát các slot, sản phẩm trùng và điều kiện thời tiết.
- Mức điểm outfit tối thiểu mặc định 6; số ứng viên mỗi slot mặc định 30; số outfit yêu cầu tối đa 5.
- Khi trời lạnh (<=18°C), hệ thống ưu tiên áo khoác nếu có ứng viên đạt điều kiện; nếu không có sẽ cảnh báo và tiếp tục ghép.
- MMR (lambda 0.25) đa dạng hóa outfit trả về. Lựa chọn màu/phong cách đầu vào là tín hiệu sở thích tường minh trong yêu cầu, không phải dữ liệu lịch sử.
- Không đủ dữ liệu tương tác rating/feedback nên không có luật cá nhân hóa dựa trên mức độ yêu thích đã học.

Đối chiếu các giá trị có thể cấu hình trong `api/config.py`; các điều kiện nghiệp vụ chi tiết trong `api/recommender.py` và `api/rules.py`.
