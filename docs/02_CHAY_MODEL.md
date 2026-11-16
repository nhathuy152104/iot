# 02 — Cài và chạy model trên máy người nhận

Các lệnh PowerShell chạy **trong folder `BAN_GIAO_MODEL`**, không phụ thuộc ổ D: hoặc đường dẫn máy cũ.

## 1. Tạo môi trường

Runtime gốc ghi Python **3.15.2**; phiên bản thư viện trong `requirements.txt`. Ưu tiên dùng cùng phiên bản để tái lập. Nếu không cài được bộ phiên bản đã khóa, không tự thay bằng bản bất kỳ rồi coi là môi trường tương đương; ghi nhận và kiểm thử sai khác.

```powershell
python --version
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r .\requirements.txt
```

Đây là môi trường nghiên cứu, không phải môi trường backend Python 3.12. Không cần GPU/CUDA. Cần mạng để tải thư viện hoặc wheelhouse được chuẩn bị riêng; bộ bàn giao chưa chứa installer/wheel offline.

## 2. Kiểm tra tính toàn vẹn và mẫu giả

```powershell
& ".\.venv\Scripts\python.exe" .\tools\verify_manifest.py
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
& ".\.venv\Scripts\python.exe" .\runtime\predict_depth.py --input .\examples\raw_synthetic.json --station synthetic-demo
```

So kết quả với `examples/expected_synthetic.json`, chấp nhận sai số số học nhỏ (rtol/atol 1e−6). Đây là mẫu GIẢ để chứng minh code/model chạy được, không là bằng chứng chất lượng dự báo hoặc dữ liệu ngập thật.

Bộ bàn giao không chứa CSV dữ liệu; CLI vẫn hỗ trợ CSV do người nhận tự tải/cung cấp. Mẫu tích hợp đóng gói dùng JSON dạng mảng quan trắc có cùng các trường.

`verify_manifest.py` kiểm tra checksum của snapshot bàn giao, không sửa model và không fit.

`MANIFEST.json` là danh sách checksum, không phải chữ ký số chống giả mạo. `KIEM_TRA_BAN_GIAO.json` là biên bản thực thi được loại khỏi manifest để tránh tham chiếu vòng. `tools/prepare_metadata.py` chỉ dành cho người chuẩn bị snapshot mới, không chạy lại để “sửa” một lỗi checksum chưa hiểu nguyên nhân.

`tools/verify_portability.py` tạo một bản sao tạm chỉ từ manifest, chạy kiểm tra checksum và mẫu giả ở đường dẫn mới rồi dọn thư mục tạm. Không cần project gốc; đây là smoke test đổi đường dẫn trên cùng môi trường, không thay việc cài và kiểm thử trên máy người nhận.

## 3. Chạy CSV raw của mình

CSV có `time`, `depth_raw_mm`; nên có `deployment_id`:

```text
time,deployment_id,depth_raw_mm
2026-11-11T10:01:00Z,station-a,70
...
```

```powershell
& ".\.venv\Scripts\python.exe" .\runtime\predict_depth.py --input "C:\data\raw.csv" --station station-a --as-of "2026-11-11T10:15:00Z"
```

- `--station`: chỉ chọn một deployment; không trộn nhiều trạm vào một chuỗi. Nếu CSV không có cột deployment, tham số này là tên người gọi tự khai báo, không xác minh nguồn.
- `--as-of` phải có timezone. Nếu bỏ qua, dùng timestamp cuối của CSV để chạy **offline**, không ngầm coi dữ liệu cũ là mới.
- Chỉ bin đã kết thúc được dùng. Nếu cutoff 10:16 thì mốc quyết định là 10:15; mẫu 10:16 không vào feature 10:15.
- Không đủ ba bin lịch sử/last-row raw null → JSON `status: unavailable`, `reason`; không fill hoặc dựng dự báo giả.
- Model/cấu hình mặc định lấy tương đối trong folder `model/`. Feature/đơn vị/hash được kiểm tra; model không nhận ID trạm/tọa độ làm feature.
- JSON trả `predicted_delta_cm`, `predicted_depth_cm`, `persistence_depth_cm`, target bin và nhãn `OFFLINE_RESEARCH_ONLY`.

Đừng gọi trực tiếp `model.predict([[8]])`: đầu vào cần 12 feature đúng thứ tự và đầu ra còn phải cộng độ sâu hiện tại. CSV ESP32 cm phải đổi thành mm trước, đồng thời kiểm tra mốc nền/hiệu chuẩn/chu kỳ đo; không mặc định giống FloodNet.

## 4. Chạy kiểm tra đầy đủ bằng project gốc (tùy chọn)

Nếu folder bàn giao vẫn nằm trong `IOT training` và các cache/cửa sổ nguyên bản còn đủ:

```powershell
& ".\.venv\Scripts\python.exe" .\tools\verify_against_project.py --training-root ".."
```

Lệnh đọc validation và toàn bộ lịch sử cache raw của các trạm validation trước 07/2023 để đối chiếu feature/prediction với 92.215 cửa sổ cũ. Không nạp cửa sổ train để fit và không đọc test. Phải giữ prefix lịch sử gốc để tái lập đúng trạng thái số học của rolling std; chỉ cắt lấy tháng validation có thể tạo sai khác số thực rất nhỏ. Trên máy người nhận thiếu dữ liệu thì lệnh này sẽ dừng rõ ràng; không đồng nghĩa model không nạp được.

Kiểm tra tổng hợp, tùy chọn ghi biên bản (chỉ file `KIEM_TRA_BAN_GIAO.json`):

```powershell
& ".\.venv\Scripts\python.exe" .\tools\check_package.py
# Nếu có cây dữ liệu gốc và muốn kiểm tra đối chiếu + ghi biên bản:
& ".\.venv\Scripts\python.exe" .\tools\check_package.py --training-root ".." --record
```
