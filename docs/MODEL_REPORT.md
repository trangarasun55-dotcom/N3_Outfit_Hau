# Báo cáo đánh giá mô hình

## Quy trình

Dữ liệu `DuLieu_ML` có 6.000 dòng, 1.500 `ID_SP`, 9 đặc trưng và không có ô thiếu. Nhãn `Điểm phù hợp (0–10)` được tạo bởi quy tắc, không phải đánh giá thực tế của người dùng. Không có trường thời gian nên chia theo nhóm sản phẩm là cách tránh để cùng sản phẩm xuất hiện ở nhiều tập. `GroupShuffleSplit` dùng seed 42, mục tiêu 70/15/15: Train 4.196 dòng/1.049 sản phẩm; Validation 904/226; Test 900/225. Metadata xác nhận các ID nhóm không giao nhau.

RF được chọn cấu hình bằng MAE Validation; các cấu hình gọn chỉ được cân nhắc trong giới hạn sai số Validation. KNN dùng GroupKFold(5) trên Train. Trọng số lai được dò bằng dự đoán OOF trên Train và bắt buộc mỗi nhánh có trọng số dương. Chọn mô hình cuối và ngưỡng tin cậy chỉ dùng Train/Validation. Sau đó fit mô hình cuối trên Train + Validation và đánh giá Test đúng một lần.

## Kết quả Test cuối cùng

| Mô hình | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mốc đoán trung bình | 2.2890 | 2.6535 | -0.0001 |
| KNN (K=3) | 0.6764 | 1.0413 | 0.8460 |
| RF | 0.2863 | 0.4536 | 0.9708 |
| Lai (RF=0.95, KNN=0.05) | 0.2875 | 0.4435 | 0.9721 |

Mô hình được chọn để xuất là `Lai`, theo ràng buộc yêu cầu cả RF và KNN thực sự tham gia; trọng số OOF là RF 0.95/KNN 0.05. RF có MAE Test nhỏ hơn một chút; Test không được dùng để thay đổi quyết định. Trên Validation, MAE Lai 0.3111 và RF 0.3116.

## Đánh giá gợi ý

Do snapshot không có phản hồi/đánh giá outfit phù hợp làm ground truth, Precision/Recall/MAP/NDCG theo nhãn workbook chỉ được báo cáo là **proxy**: mức đồng thuận xếp hạng với nhãn quy tắc `>=6/10`, trên các nhóm sản phẩm giữ lại theo hoàn cảnh và nhiệt độ. Không diễn giải các số này là sở thích người dùng. Đánh giá hệ thống trên 20 bối cảnh tổng hợp: diversity trung bình 1.0 theo `1 - Jaccard(ID sản phẩm)`, catalog coverage 95/1.500 (6,33%); 3/20 bối cảnh trả ít hơn 3 outfit. Đây không phải lưu lượng thật. Chi tiết có trong `artifacts/evaluation/`.

## Giới hạn

- R² cao chủ yếu cho thấy mô hình học lại công thức sinh nhãn, không xác nhận khả năng hiểu gu thời trang.
- Không có log tương tác đủ để huấn luyện/đánh giá Collaborative Filtering; hệ thống dùng Content-Based Filtering cùng RF–KNN, quy tắc ghép và đa dạng hóa DSS.
- Độ lệch chuẩn giữa các cây RF là chỉ báo phân tán, không phải khoảng tin cậy thống kê. Ngưỡng mức tin cậy lấy từ phân vị Validation.
- Chưa đo tải HTTP/đồng thời hoặc thử nghiệm A/B với người dùng.
- Phiên bản thư viện và hash tệp được ghi trong metadata/manifest; cần môi trường tương thích với `requirements.txt`.
