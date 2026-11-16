# Bàn giao training và model dự báo độ sâu ngập

**Bắt đầu tại đây.** Bộ bàn giao cục bộ ngày 11/11/2026, không phải model đã được duyệt vận hành. Không tự gửi bộ này cho bên thứ ba trước khi kiểm tra quyền FloodNet, kể cả trọng số và artifacts phái sinh.

## Đọc theo thứ tự

1. [Tổng quan và model đang chọn](docs/01_TONG_QUAN.md).
2. [Cài đặt và chạy dự đoán](docs/02_CHAY_MODEL.md).
3. [Dữ liệu và train tiếp](docs/03_DU_LIEU_VA_TRAIN_TIEP.md).
4. [Checklist nghiệm thu](docs/04_CHECKLIST.md).
5. `reference/`: bản sao tài liệu/báo cáo tiến trình để đọc sâu hơn; không cần đọc session JSON để bắt đầu.

## Trong folder có gì?

```text
BAN_GIAO_MODEL/
  README.md
  requirements.txt               # Môi trường nghiên cứu đã ghi nhận
  docs/                          # Hướng dẫn bàn giao
  model/                         # Model đang chọn + cấu hình/chỉ số gốc
  runtime/predict_depth.py        # CSV raw -> 12 feature -> delta -> độ sâu
  examples/raw_synthetic.json     # Dữ liệu GIẢ, không phải FloodNet/đánh giá chất lượng
  examples/expected_synthetic.json
  tests/                         # Test inference và checksum model
  tools/                         # Kiểm tra checksum, đối chiếu với project gốc
  training_source/analysis/       # Snapshot code nghiên cứu, không gồm dữ liệu/.venv
  reference/                     # Snapshot tài liệu và báo cáo kết quả
  DATA_INVENTORY.json             # Các file dữ liệu gốc cần bổ sung, không chứa quan trắc
  MANIFEST.json                   # SHA256 các file trong bộ bàn giao
  KIEM_TRA_BAN_GIAO.json           # Kết quả kiểm tra đã thực hiện
```

**Chỉ chạy model:** folder này tự chứa model/code và mẫu giả để kiểm tra. Không cần backend, Docker, MQTT, frontend hoặc bộ CSV 4,46 GB.

**Muốn train từ dữ liệu tự tải:** có `tools/train_from_download.py`, chạy một cấu hình raw ba mã cố định từ CSV tải về, không cần cache/run cũ. **Muốn tái lập đối chiếu toàn bộ nghiên cứu cũ:** cần thêm cây dữ liệu/cửa sổ/cache gốc; xem tài liệu số 03. Folder này không giả vờ chứa toàn bộ dữ liệu training.

**Không kèm bất kỳ file CSV nguồn, cache hoặc cửa sổ quan trắc nào.** Người nhận tự tải dữ liệu từ link được cung cấp. Các bảng metric nhỏ được lưu JSON; ví dụ kiểm tra là JSON hoàn toàn giả. Link nguồn: <https://zenodo.org/records/10211443>.

Không mang `.venv` từ máy cũ sang máy mới; tạo môi trường mới. Bản model dùng cho bản đồ hiện tại là **raw ba mã `raw_delta_d3_weight20`**, không phải model mười mã hoặc trial absolute-error.
