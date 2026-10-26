# Kiến trúc hiện tại

Khảo sát theo mã trong repo ngày 10/10/2026; đây là mô tả hiện trạng, không phải kiến trúc mục tiêu sau các giai đoạn tiếp theo.

## Thành phần và giao thức

- **Web là Flask**, khởi tạo trong `web/app.py`; các trang được render bằng Jinja2 từ `web/templates/`.
- **API là FastAPI**, khởi tạo trong `api/main.py`.
- `web/` **gọi API qua HTTP** bằng `requests` (`API_URL`, mặc định `http://127.0.0.1:8000`). Web không import trực tiếp module `api/`. JavaScript gọi các route nội bộ Flask như `/ui/weather`; Flask mới chuyển tiếp sang API.
- FastAPI nạp metadata/catalog và `model/outfit_model.joblib` ở cấp module khi khởi động. Thời tiết Open-Meteo và Gemini (nếu được cấu hình) chỉ được gọi từ backend API.
- SQLite lưu yêu cầu gợi ý, từng outfit, lượt xem, lượt chốt và đánh giá. Các endpoint lịch sử/hành vi/admin xác thực bằng cookie phiên HttpOnly ký ở API; quyền admin được kiểm tra phía server. `/recommend` vẫn nhận `X-User-Id` theo hợp đồng Flask cũ, không dùng header đó để cấp quyền đọc lịch sử hành vi.

## Luồng một request gợi ý

```text
Người dùng / trình duyệt
  │ GET /
  ▼
Flask web/app.py
  ├─ requests.get(API_URL/model-info) ───────────────────────────────┐
  ├─ trình duyệt gọi /ui/weather → Flask proxy → GET API_URL/weather │
  ├─ POST /activity → ghi hoạt động/giới tính/nhiệt độ vào session    │
  └─ POST /style → ghi phong cách/màu/formality vào session           │
                                                                        ▼
                                                              FastAPI api/main.py
                                                                ├─ Pydantic validate
                                                                ├─ recommender đọc catalog
                                                                ├─ Pipeline/model đã nạp
                                                                ├─ giải thích Gemini tùy chọn
                                                                └─ SQLite lưu request/history
                                                                        │ JSON
                                                                        ▼
Flask web/app.py ◀──── POST /recommend (requests, X-User-Id) ◀──────────┘
  ├─ lưu kết quả trong dict OUTFIT_RESULTS trong bộ nhớ tiến trình
  ├─ lưu result_id trong session cookie
  └─ redirect /results → Jinja2 render Top-N → /detail/<id>
```

### Tính bền vững theo thành phần

- DB hiện có `requests` và `feedback`; kết quả lịch sử có thể nạp lại từ API/SQLite.
- Kết quả dùng ngay trên `/results` và `/detail` được giữ trong dict cấp tiến trình Flask, không phải SQLite; restart hoặc nhiều worker có thể làm trang kết quả không thấy object tương ứng.
- Profile và schedule hiện nằm trong Flask session; chưa có bảng/CRUD tương ứng trong API/DB.

## Cập nhật Giai đoạn 3 — recommender

- `POST /recommend` xác thực request bằng Pydantic, lọc gender và khoảng nhiệt độ món, dự đoán theo batch từ artifact đã nạp lúc import, rồi ghép Top-N bằng điểm model. Phong cách/màu/formality chỉ tạo ưu tiên xếp hạng; ma trận `rules.py` chỉ phục vụ diễn giải.
- Mặc định ngưỡng điểm model tối thiểu là 6/10, biên ứng viên khởi đầu là 1 điểm so với bộ tốt nhất (nới từng 0,25 khi cần), và MMR dùng λ=0,25. Không hai bộ nào dùng chung món chính; khi không đủ ứng viên cả sau khi nới biên, kết quả ít hơn được kèm cảnh báo.
- Gemini nhận một batch cho các outfit chưa có cache, trả JSON có schema và được kiểm tra số liệu/evidence, món, tiếng Việt, confidence và độ tương đồng. Không có key, lỗi dịch vụ hoặc JSON không hợp lệ thì dùng nhận xét quy tắc theo hạng, không làm hỏng phản hồi gợi ý.
- Migration SQLite v2 tạo `recommendation_requests` và `recommendation_items`, đồng thời giữ việc ghi legacy `requests` để lịch sử/phản hồi hiện tại tiếp tục hoạt động. Mỗi transaction ghi context request và các outfit đã nhận xét.

```text
POST /recommend
  → validate request
  → lọc catalog theo giới tính + Tmin/Tmax
  → batch-score Pipeline/RF (+ KNN nếu artifact chọn Lai)
  → ghép outfit → MMR và kiểm tra món chính không trùng
  → Gemini JSON (batch/cache) hoặc fallback rules
  → SQLite: recommendation_requests + recommendation_items + requests
  → JSON Top-N, confidence, decision-matrix và explanation
```

## Hành vi, lịch sử và thống kê admin

- Migration SQLite v3 thêm `outfit_views`, `finalized_outfits` và `outfit_ratings`; đánh giá có `UNIQUE(user_id, request_item_id)` nên đánh giá mới thay đánh giá cũ. Mỗi kết quả outfit trả thêm `request_item_id`, dùng làm `{id}` cho các route hành vi.
- `POST /events/view` nhận `start`, `heartbeat`, `end`; thời lượng đo từ thời điểm bắt đầu ở server, heartbeat cập nhật trạng thái, và tự kết thúc khi chạm trần 1.800 giây. Không nhận thời lượng do client tự khai.
- `POST /outfits/{id}/finalize` chốt; `DELETE` cùng route hủy chốt. `POST /outfits/{id}/feedback` lưu giá trị `1` hoặc `-1`.
- `GET /history` chỉ dùng session đã xác thực, phân tab `viewed|finalized|feedback` và range `day|week|month|all|from-to`. Ngày/tuần/tháng được cắt theo nửa mở `[đầu ngày, đầu ngày kế tiếp)` tại `Asia/Ho_Chi_Minh`, sau đó chuyển sang UTC; timestamp mới lưu dạng ISO 8601 có hậu tố `Z`.
- `/admin/stats`, `/admin/users`, `/admin/outfits`, `/admin/export.csv` yêu cầu role admin. Thống kê hệ thống không được trả cho user thường. Dữ liệu CSV là bảng tổng hợp outfit gồm số gợi ý/xem, thời lượng, chốt và đánh giá.

## Ghi chú khảo sát môi trường

- Python dùng để chạy baseline: 3.14.7.
- DB `data/app.db` tại thời điểm khảo sát có 14 dòng `requests`, 0 dòng `feedback`.
- `web/` là client REST Flask; không nạp model. Model được nạp tại API khi import module.

## Đăng nhập và đăng xuất (web)

- Chỉ `/login`, `/register`, `/healthz`, `/favicon.ico` và `/static/*` là công khai. Mọi route khác đi qua `require_login` trong `web/app.py`: chưa có phiên thì chuyển về `/login` (các route `/ui/*` trả JSON 401).
- Trang đăng nhập/đăng ký dùng layout riêng `auth_base.html` (không có thanh điều hướng). Nút **Đăng xuất** (POST `/logout`) nằm ở cuối thanh bên trái, xóa phiên web, gọi `/auth/logout` của API rồi chuyển về `/login`.
- Trang đã đăng nhập gửi `Cache-Control: no-store` để nút Back không hiện lại nội dung sau khi đăng xuất.
