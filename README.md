# OutfitAI

Đồ án Hệ trợ giúp ra quyết định hỗ trợ sinh viên kiến trúc chọn và phối trang phục theo hoạt động, thời tiết và sở thích. Ứng dụng gồm pipeline huấn luyện scikit-learn, API FastAPI và web client Flask.

## Kiến trúc

```text
data/DatasetDSS.xlsx (nguồn dữ liệu chuẩn)
        │ model/train_export.py
        ├── model/outfit_model.joblib + model/model_metadata.json
        └── data/mon_do.csv (artifact danh mục được sinh tự động)
                       │
                 FastAPI (api/)
            SQLite · Open-Meteo · Gemini tùy chọn
                       │ REST / cookie phiên
                    Flask (web/)
                       │
                 Trình duyệt
```

Flask không nạp model và gọi API qua `API_URL`; các tích hợp Open-Meteo và Gemini chỉ chạy trong FastAPI. API nạp pipeline, metadata và danh mục món một lần khi khởi động. `DatasetDSS.xlsx` là nguồn sự thật duy nhất cần chỉnh sửa; không sửa `data/mon_do.csv` bằng tay vì script huấn luyện sẽ xuất lại file này.

## Yêu cầu và cài đặt

Python 3.12–3.14; các phiên bản thư viện được ghim trong `requirements.txt`. Các lệnh dưới đây dùng một virtualenv mới ở thư mục tạm để không để `.venv` trong repository.

### Windows PowerShell

Chạy tại thư mục gốc dự án:

```powershell
$env:OUTFITAI_VENV = Join-Path $env:TEMP "outfitai-venv"
py -3.14 -m venv $env:OUTFITAI_VENV
& "$env:OUTFITAI_VENV\Scripts\python.exe" -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
& "$env:OUTFITAI_VENV\Scripts\python.exe" model\train_export.py
```

Nếu không có `py -3.14`, thay bằng `python -m venv $env:OUTFITAI_VENV` và dùng phiên bản Python tương thích đã cài. Không ghi API key vào mã nguồn, lệnh shell có lưu lịch sử, hoặc commit; mở `.env` cục bộ để tự cấu hình. Có thể bỏ `GEMINI_API_KEY`; giải thích theo quy tắc vẫn hoạt động.

### Linux / macOS

```bash
export OUTFITAI_VENV="${TMPDIR:-/tmp}/outfitai-venv"
python3 -m venv "$OUTFITAI_VENV"
"$OUTFITAI_VENV/bin/python" -m pip install -r requirements.txt
test -f .env || cp .env.example .env
"$OUTFITAI_VENV/bin/python" model/train_export.py
```

Model được lưu trong `model/`, biểu đồ trong `model/charts/`, và `data/mon_do.csv` được xuất từ sheet `MonDo`. Không xóa/sửa workbook gốc. Nếu scikit-learn báo không tương thích với joblib đã export, cài đúng pin trong requirements và chạy lại lệnh train.

## Chạy ứng dụng

### Một lệnh (khuyến nghị)

Sau khi kích hoạt venv hoặc gọi Python bằng đường dẫn đầy đủ:

```powershell
& "$env:OUTFITAI_VENV\Scripts\python.exe" run_dev.py
```

Lệnh này khởi động API, đợi `/health` sẵn sàng rồi mới mở web. Truy cập `http://127.0.0.1:5000`: trang đầu tiên là **Đăng nhập / Đăng ký** (chưa đăng nhập thì mọi trang khác đều chuyển về đây). Sau khi đăng nhập, nút **Đăng xuất** nằm ở góc dưới bên trái thanh điều hướng và đưa bạn về trang đăng nhập. Swagger UI tại `http://127.0.0.1:8000/docs`. Dừng cả hai bằng `Ctrl+C`. Tùy chọn cổng: `run_dev.py --api-port 8001 --web-port 5001`.

### Chạy tách tiến trình

Mở hai terminal tại thư mục gốc:

```powershell
# Terminal 1 — API
& "$env:OUTFITAI_VENV\Scripts\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — web
& "$env:OUTFITAI_VENV\Scripts\python.exe" web\app.py
```

Trong Linux/macOS, thay `Scripts\python.exe` bằng `$OUTFITAI_VENV/bin/python` và `web\app.py` bằng `web/app.py`.

## Cấu hình

Tạo `.env` từ `.env.example` nếu chưa có. Không hiển thị hoặc chia sẻ nội dung `.env`.

| Biến | Mục đích |
|---|---|
| `API_URL` | URL API mà Flask gọi; mặc định `http://127.0.0.1:8000` |
| `SECRET_KEY` | Khóa ký cookie của web và API. Để trống thì hệ thống tự tạo và lưu ở `data/.secret_key` (phiên không mất khi khởi động lại) |
| `GEMINI_API_KEY` | Tùy chọn; chỉ FastAPI đọc. Thiếu key thì dùng fallback theo quy tắc |
| `GEMINI_MODEL` | Tên model Gemini chính; API không tự đoán tên nếu biến này trống. `.env.example` có giá trị mẫu `gemini-3.1-flash-lite`. Khi model chính trả 404/5xx hoặc câu trả lời không đạt kiểm tra dữ kiện, API lấy danh sách model hiện có từ Gemini, lọc model Flash hỗ trợ `generateContent`, rồi thử tối đa một model dự phòng. Danh sách được lưu tạm 30 phút. Lỗi khóa/quyền, quota, kết nối hoặc dịch vụ ngừng hoạt động vẫn có thể khiến API dùng giải thích theo quy tắc. FastAPI ưu tiên biến môi trường tiến trình, sau đó mới nạp `.env` |
| `ADMIN_EMAIL`, `ADMIN_PASSWORD` | Seed tài khoản admin khi API khởi động; không có tài khoản/mật khẩu mặc định công khai |
| `DB_PATH` | SQLite; mặc định `data/app.db` |
| `DEFAULT_CITY`, `TEMP_MODE` | Địa điểm mặc định và cách chọn nhiệt độ cho model |
| `COOKIE_SECURE`, `CORS_ORIGINS`, `FLASK_DEBUG` | Cấu hình cookie, CORS và chế độ debug |

Tài khoản admin được seed từ `ADMIN_EMAIL` và `ADMIN_PASSWORD` ở lần khởi động API. Đặt hai biến này trước khi khởi động lần đầu; dùng mật khẩu mạnh và không dùng thông tin mẫu trên hệ thống thật. Session của API dùng cookie HttpOnly. Không có chế độ khách: phải đăng nhập mới dùng được gợi ý, lịch sử, hồ sơ và trang admin. Sau khi đăng ký, đăng nhập lại bằng tài khoản vừa tạo.

## Kiểm thử và báo cáo kịch bản

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:PYTHONIOENCODING = 'utf-8'
& "$env:OUTFITAI_VENV\Scripts\python.exe" -m pytest -q -p no:cacheprovider --basetemp=.pytest-tmp\full
& "$env:OUTFITAI_VENV\Scripts\python.exe" tests\run_api_scenarios.py
```

`run_api_scenarios.py` dùng TestClient và SQLite tạm, không gọi dịch vụ Gemini bên ngoài; xuất bảng trường hợp, kết quả mong đợi/thực tế và thời gian phản hồi. Cần `PYTHONPATH` để script tìm được package `api`; `PYTHONIOENCODING` giúp Windows in tiếng Việt. Kết quả mới nhất được in trực tiếp khi chạy, không lấy số liệu cũ làm cam kết thời gian trên máy khác.

Để kiểm tra UI thật, khởi động `run_dev.py`, mở trình duyệt, rồi chạy luồng: đăng ký user → đăng nhập → gợi ý → xem → chốt → đánh giá → lịch sử → đăng xuất → đăng nhập admin → xem thống kê. Kiểm tra nhanh các kết nối (API, thời tiết, Gemini gọi thật) khi API đang chạy: `python scripts/check_connections.py`. Bài kiểm tra Gemini gọi model được cấu hình và chỉ in trạng thái/mã lỗi, không in khóa; cần cung cấp `GEMINI_API_KEY` và `GEMINI_MODEL` trong môi trường của tiến trình chạy script.

## API và dữ liệu

- `GET /health`, `GET /model-info`, `POST /predict`, `POST /recommend`
- `GET /weather`, `GET /geocode`
- `GET /profile`, các endpoint xác thực `/auth/*`
- `GET /history`, `POST /events/view`, chốt/hủy chốt và đánh giá outfit
- `/admin/stats`, `/admin/users`, `/admin/outfits`, `/admin/export.csv` (admin-only)

SQLite được tạo/migrate tự động; không commit `data/app.db`. Gợi ý, lượt xem, chốt và đánh giá được lưu theo user. Mốc thời gian lưu UTC ISO 8601 có `Z`; bộ lọc lịch sử dùng múi giờ `Asia/Ho_Chi_Minh`.

## Giới hạn đã biết

- Nhãn huấn luyện được sinh từ công thức, nên điểm model cao không chứng minh mức độ hài lòng thực tế.
- Lịch học bền vững, CRUD lịch và đọc ảnh lịch chưa được triển khai hoàn chỉnh; không dùng dữ liệu lịch làm đặc trưng model.
- Nếu thiếu Gemini key, lời giải thích quy tắc được dùng thay thế. Thời tiết cần kết nối Open-Meteo.
- Định danh người dùng dựa trên phiên cookie, không phải hệ thống đăng nhập doanh nghiệp; bật HTTPS và `COOKIE_SECURE=1` khi triển khai.
- Kiến trúc và báo cáo mô hình: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/MODEL_REPORT.md`](docs/MODEL_REPORT.md).


## Model hiện tại

Artifact `model/outfit_model.joblib` được tạo từ workbook `data/DatasetDSS.xlsx` bằng `model/train_export.py`. Xem `docs/MODEL_REPORT.md` để biết chỉ số đánh giá cuối cùng và quy trình chia dữ liệu.

Hệ thống gợi ý dùng CBF từ thuộc tính sản phẩm và sở thích nhập trong yêu cầu; điểm xếp hạng dùng RF–KNN. Không có feedback/rating đủ để đánh giá relevance thực tế hoặc huấn luyện Collaborative Filtering. Báo cáo audit, kiến trúc, luật DSS, giới hạn và kết quả kiểm tra nằm trong `reports/` và `test_results/`. Chạy `python scripts/run_evaluation.py` để tái lập các bảng số liệu và manifest. Các chỉ số ranking từ nhãn workbook được đánh dấu proxy, không phải mức hài lòng người dùng.


## Đồng bộ dataset và ảnh từ `Trangphuc_DSS`

- `data/DatasetDSS.xlsx` là workbook đồng bộ từ `Trangphuc_DSS/DatasetDSS.xlsx` (các sheet `MonDo`, `DuLieu_ML` và dữ liệu liên quan được giữ nguyên).
- `data/mon_do.csv` là danh mục mà API đọc; khi sửa workbook, hãy chạy `model/train_export.py` để tái tạo danh mục và pipeline mô hình.
- Ảnh sản phẩm được đặt tại `web/static/images/images/SPxxxx.jpg` để khớp đường dẫn ảnh trong trang kết quả. Một số ID trong bộ ZIP nguồn không có ảnh; giao diện dùng placeholder thay vì làm hỏng trang.
- Mô hình và metadata được lưu tại `model/outfit_model.joblib` và `model/model_metadata.json`; API cần cấu trúc pipeline tương thích trong artifact.
- Kiểm tra nhanh dữ liệu/ảnh bằng `python scripts/check_dataset_assets.py`.


## Huấn luyện mô hình (đã đồng bộ với Trangphuc_DSS)

Từ thư mục gốc dự án, cài thư viện trong `requirements.txt`, sau đó chạy:

```bash
python model/train_export.py
```

Pipeline chia Train/Validation/Test theo `ID_SP` (mục tiêu 70/15/15), bảo đảm một sản phẩm chỉ thuộc một tập. Cấu hình RF và mô hình mặc định được chọn bằng Validation; KNN dùng `GroupKFold(5)` và trọng số lai được chọn từ dự đoán OOF trên Train. Ngưỡng tin cậy được hiệu chỉnh bằng Validation. Test chỉ được dùng để đánh giá cuối cùng sau khi đã chốt lựa chọn. Mô hình cuối được fit trên Train + Validation. Script xuất mô hình, metadata và 4 biểu đồ trong `model/charts/`.

Các biểu đồ phản ánh lần train gần nhất; hãy chạy lại script để cập nhật chúng cùng mô hình.
