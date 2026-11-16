# 03 — Dữ liệu, tái lập và train tiếp

## Có và không có trong bộ bàn giao

**Có:** model đang chọn, cấu hình/metric/lựa chọn gốc, runtime inference độc lập, snapshot các file Python nghiên cứu và tài liệu/báo cáo. `.venv`, CSV nguồn, cache, cửa sổ train/validation và frontend/backend không được nhân đôi.

**Không kèm CSV dữ liệu hoặc Parquet quan trắc:** khoảng 4,46 GB CSV và các file Parquet vẫn ở `IOT training` gốc. Mẫu `examples/raw_synthetic.json` hoàn toàn giả. Xem `DATA_INVENTORY.json` để biết đường dẫn tương đối, số hàng/checksum đã ghi nhận của cửa sổ; CSV nguồn được kiểm kê theo size/mtime, không được mô tả nhầm là đã hash đầy đủ.

Người nhận tự tải từ **<https://zenodo.org/records/10211443>**, hoặc link người giao cung cấp, đọc/chấp thuận điều khoản nguồn. Dùng đúng phiên bản/file tháng và tên file trong inventory, không tự thay bằng bộ ảnh FloodNet hoặc phiên bản khác rồi coi là tái lập. Link tải không đồng nghĩa giấy phép tái phân phối model đã được xác nhận.

## Nếu người nhận chỉ cần inference

Chỉ cần bộ này + môi trường thư viện và raw đầu vào hợp lệ. Không cần tải lại FloodNet để chạy mẫu giả. Khả năng chạy được khác với việc đã kiểm định trên dữ liệu trạm của họ.

## Nếu người nhận cần train tiếp/tái lập nghiên cứu

### Cách A — Chỉ tải CSV từ web, không cần cache/run cũ

Có entry point độc lập đọc CSV theo tháng từ folder đã giải nén. Chỉ chọn ba deployment cũ, chỉ đọc các file trước 07/2023, dựng feature/nhãn/split theo chính sách cũ. Một cấu hình **d3/250 cây/weight20/seed42**, không tự quét tham số hoặc promote model.

```powershell
# Tất cả lệnh chạy trong BAN_GIAO_MODEL. Thư mục nguồn là dữ liệu người nhận tự tải.
& ".\.venv\Scripts\python.exe" .\tools\train_from_download.py --data-dir "C:\FloodNet-download" --output-dir "C:\AI-new-runs" --prepare-only

# Sau khi kiểm tra counts/provenance, chạy lượt mới để fit và đánh giá validation.
& ".\.venv\Scripts\python.exe" .\tools\train_from_download.py --data-dir "C:\FloodNet-download" --output-dir "C:\AI-new-runs"
```

Script yêu cầu đủ các file tháng trước test theo inventory; file test từ 07/2023 trở đi không được mở. Nó ghi SHA256 CSV thực sự dùng, tạo output timestamp mới, kiểm tra số cửa sổ **270.312 train / 92.215 validation** trước fit; khác thì dừng để đối soát nguồn, không âm thầm hạ tiêu chí. Không ghi đè model trong `model/`. Không lưu toàn bộ CSV/cửa sổ vào bộ bàn giao.

Đây là đường chạy thuận tiện cho người nhận, **không thay snapshot code lịch sử**. Khi chuẩn bị bàn giao đã unit-test phần chuẩn bị và đối chiếu membership/feature/nhãn validation với dự án gốc; chưa chạy lại toàn bộ 33 CSV/fit bằng CLI này để tiết kiệm xử lý dữ liệu lớn. Người nhận cần chạy `--prepare-only` và nghiệm thu trên máy họ trước khi coi là tái lập end-to-end. Phiên bản nguồn web mới có thể khác archive cũ dù cùng tên; đối chiếu nguồn/hash trước khẳng định kết quả.

### Cách B — Tái lập các thí nghiệm/báo cáo lịch sử đầy đủ

Ưu tiên làm trên **bản sao đầy đủ có phép sử dụng của cây `IOT training`**, giữ cấu trúc:

```text
IOT training/
  floodnet-flood-sensor-*.csv
  audit/
  analysis/
    five_minute/selection_manifest.json
    five_minute/local_cache/
    five_minute/provisional_windows/
    training_runs/<các run có provenance và cửa sổ>
    raw_branch/, raw_expansion/, các script train/validate
```

`training_source/analysis` là **snapshot code để đọc/khôi phục**, không phải lời hứa chạy train ngay khi thiếu cây dữ liệu/artifacts ở trên. Không chép đè snapshot lên project đang có thay đổi mà chưa đối chiếu.

Code gốc thường suy ra root từ vị trí script. Provenance cũ có đường dẫn `D:\CODE\IOT training`; `analysis/run_relocated.py` ánh xạ đường dẫn JSON trong bộ nhớ theo root script hiện tại, không sửa file đã đóng băng hoặc bỏ checksum. Những đường dẫn tuyệt đối từ các lần chạy ở root khác có thể cần khai báo migration mới; không thay hàng loạt rồi bỏ kiểm tra hash.

## Trình tự tái lập đề xuất

1. Kiểm kê dữ liệu/quyền sử dụng, kiểm tra đủ model/config/cửa sổ theo manifest.
2. Chạy test `analysis/`, rồi test `analysis/five_minute/` theo `reference/analysis/TRAINING_README.md`.
3. Nạp model cũ, tái lập validation; chưa fit hoặc mở test chỉ vì đọc tài liệu.
4. Đọc các kết quả raw, mở rộng, phân tích A, cảnh báo B, trial objective MAE. Không lặp lại những bước đã có mà không có giả thuyết mới.
5. Muốn train mới: chốt giả thuyết, tiêu chí, tập dữ liệu và cấu hình trước fit; script tạo run mới, không ghi đè run cũ.

Các entry point chính trong snapshot:

| Script | Vai trò |
|---|---|
| `train_raw_delta.py` | Nhánh raw ba mã, bốn cấu hình, nhãn proc tương lai |
| `train_raw_expanded.py` | Thử bổ sung bảy deployment, giữ validation cũ |
| `validate_raw.py`, `validate_raw_expanded.py` | Tái lập run đã có; có thể ghi lại báo cáo/validation artifact |
| `diagnose_raw_expansion.py` | Phần A: phân tích khác biệt và sai số |
| `evaluate_warning_events.py` | Phần B: đánh giá theo đợt |
| `depth_accuracy.py` | Trial cố định đổi objective sang absolute error |
| `validate_depth_accuracy.py` | Tái lập trial, chỉ đọc |
| `export_forecast_replay.py` | Export local cho frontend `/forecast`, không live |

Không chạy `train_baseline_xgboost.py`/`diagnose_training.py` tùy tiện nếu yêu cầu là không đọc test: lượt đầu/chẩn đoán cũ có thể đọc test cũ. Các validator cũ cũng không phải tất cả đều chỉ đọc. Đọc docstring/README trước chạy.

## Việc tiếp theo chưa hoàn thành

- Chốt tập xác nhận độc lập trước chọn model tiếp; validation cũ đã dùng nhiều lần.
- Nghiên cứu lịch sử dài hơn/feature bền vững với nhiễu trong thí nghiệm riêng, không thay nhiều yếu tố cùng lúc.
- Kiểm định raw/proc và độ sâu thực, đánh giá chuyển miền sang ESP32.
- Nếu muốn khoảng dự báo: hiệu chuẩn/kiểm định thích hợp, không lấy MAE làm ± giới hạn bảo đảm.
- Chưa duyệt AI cho cảnh báo, định tuyến hoặc vận hành thực tế.

## Giấy phép/bảo mật

FloodNet NYU/CUNY: nguồn công khai không có nghĩa tự do tái phân phối. Điều khoản người dùng cung cấp giới hạn phi thương mại, không chia sẻ CSV gốc kể cả một phần cho bên thứ ba nếu chưa có chấp thuận. Cần làm rõ quyền với trọng số/artifacts phái sinh trước gửi bộ này. Người nhận có thể cần tự lấy dữ liệu và chấp thuận thỏa thuận nguồn.

Không đưa `.env`, khóa MQTT/ORS, mật khẩu, session JSON hoặc PDF không liên quan vào bộ bàn giao. Bộ này được lắp từ danh sách file cho phép, không copy nguyên workspace.
