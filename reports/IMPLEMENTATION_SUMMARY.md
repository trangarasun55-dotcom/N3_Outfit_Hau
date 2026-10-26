# Tổng kết triển khai

- Chia Train/Validation/Test theo ID_SP, không trùng nhóm.
- Không dùng Test để chọn RF, KNN, mô hình, trọng số lai hay ngưỡng độ tin cậy.
- KNN K=3 theo GroupKFold(5); trọng số lai RF 0.95/KNN 0.05 theo OOF Train.
- Huấn luyện artifact cuối trên Train + Validation; cập nhật `model_metadata.json` và artifact liên quan.
- Sửa kiểm tra roundtrip theo đúng model name; giữ rõ các nhánh RF/KNN/Lai/mốc trung bình.
- Tích hợp CBF dựa trên hồ sơ đầu vào tường minh, kết hợp luật ghép và đa dạng hóa DSS.
- Gỡ template đánh giá AI mồ côi; endpoint/đường dẫn trang đánh giá không được phục vụ.
- Bổ sung script đánh giá tái lập, metrics, manifest/hash, data profile, báo cáo kiến trúc, audit, test và giới hạn.
- Loại bỏ `overflow-x:hidden` toàn cục; giữ các breakpoint và quy tắc responsive hiện có.
