# Quy trình huấn luyện và xuất model

`model/train_export.py` dùng split theo `ID_SP` (GroupShuffleSplit, seed 42; mục tiêu 70/15/15), kiểm tra không giao nhau và chỉ lựa chọn trên Train/Validation.

- Cấu hình RF được chọn bằng MAE Validation. KNN được chọn bằng GroupKFold(5) trên Train.
- Trọng số RF–KNN được lựa chọn trên dự đoán OOF của Train; ứng dụng xuất mô hình lai với cả hai trọng số dương.
- Mô hình mặc định và ngưỡng độ tin cậy chỉ được hiệu chỉnh bằng Train/Validation.
- Sau khi khóa lựa chọn, mô hình cuối được fit trên Train + Validation. Test chỉ đánh giá cuối cùng.
- `verify_fresh_process()` kiểm tra dự đoán của RF, KNN, Lai và mốc trung bình theo đúng mô hình được chọn.

Metadata chứa split, ID sản phẩm từng phần, chỉ số Validation/Test, cấu hình, trọng số OOF, ngưỡng tin cậy, phân tích lỗi và roundtrip. Artifact gồm `model/outfit_model.joblib`, `model/model_metadata.json`, `data/mon_do.csv` và biểu đồ trong `model/charts/`.

Chạy từ thư mục gốc:

```bash
python model/train_export.py
```

Chạy đánh giá tái lập sau khi huấn luyện:

```bash
python scripts/run_evaluation.py
```

Tập Test không được dùng để chọn bất kỳ cấu hình, mô hình, trọng số hoặc ngưỡng nào.
