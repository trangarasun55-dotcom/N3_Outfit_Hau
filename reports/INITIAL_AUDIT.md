# Báo cáo rà soát ban đầu

- Bản ZIP nguồn có trang template `model_evaluation.html` mồ côi nhưng không có route hoạt động; template đã được gỡ và kiểm thử đường dẫn trực tiếp trả 404.
- Luồng xuất trước đây chọn cấu hình dựa vào Test và kiểm tra tiến trình mới không xử lý đúng mọi tên mô hình. Đã chuyển quyết định sang Validation/OOF và roundtrip có nhánh cho RF, KNN, Lai, mốc trung bình.
- Chia nhóm theo `ID_SP` là cần thiết vì nhiều dòng thuộc cùng sản phẩm. Không dùng cột nhãn tập cũ làm split.
- Workbook có 6.000 dòng ML/1.500 sản phẩm và 1.500 sản phẩm trong danh mục. Nhãn điểm là nhãn quy tắc.
- Snapshot DB được rà soát ở mức thống kê: không có rating/feedback outfit; có lượt xem và request nhưng không có tương tác đánh giá phù hợp. Vì vậy chưa đủ cơ sở dùng Collaborative Filtering hay đo sở thích thật.
- Một số ghi chú cũ trong tài liệu sai với hiện trạng; đã sửa báo cáo mô hình và huấn luyện.
