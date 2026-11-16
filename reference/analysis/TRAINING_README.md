# Train hồi cứu — baseline và XGBoost

## Chạy

Thư mục hiện tại: `D:\CODE\IOT\IOT training`. Các lệnh đọc provenance cũ dùng `run_relocated.py` để ánh xạ đường dẫn JSON trong bộ nhớ, không sửa artifacts hoặc bỏ checksum. Wrapper không thay hành vi train/ghi báo cáo của script gốc; không chạy lại chỉ để đọc kết quả.

```powershell
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\train_baseline_xgboost.py"
```

CPU, tối đa 4 luồng, seed42. Mỗi lượt tạo `training_runs/<UTC timestamp>/`, không ghi đè kết quả cũ. Cần pandas/numpy/matplotlib/pyarrow/xgboost/scikit-learn; phiên bản trực tiếp trong `requirements.txt`, phiên bản runtime trong cấu hình mỗi run. Môi trường hiện dùng Python 3.15; không thay môi trường backend.

## Thí nghiệm cố định

- Dùng ba trạm `selected_core` và danh sách feature/label từ manifest 5 phút. Không dùng ID trạm, audit tương lai, cờ đối soát hoặc nhãn làm đặc trưng.
- Giữ toàn bộ cửa sổ eligible gốc; không dùng mask offline để chỉ giữ ca dễ. Không thêm bảy trạm chưa đối soát xong vào core.
- Baseline: giữ độ sâu hiện tại và ngoại suy tốc độ 5 phút vừa qua thêm 10 phút, đều giới hạn dưới 0.
- Bốn cấu hình XGBoost: depth3/250cây hoặc depth4/400cây, trọng số mẫu thường hoặc nhân20 cho nhãn train>=10cm. Không resample validation/test.
- Chọn cấu hình bằng trung bình MAE hai nhóm nhãn >=10cm và <10cm trên validation, không bằng MAE toàn ngày khô và không bằng test.
- Không refit trên validation, không early-stopping bằng test. Lưu quyết định trước nạp test và kiểm tra model lưu/nạp lại cho kết quả giống nhau.
- Không tự triển khai model ra API/frontend. XGBoost vẫn có thể kém baseline.

## Giới hạn

Model dự báo bin tại t+10 phút (số đo thực có thể nằm trong phút thứ5–10), không phải cực đại trong 10 phút hoặc độ sâu tức thời chính xác tại phút thứ10. Các cửa sổ có tương quan và số cửa sổ vượt ngưỡng không phải số sự kiện độc lập.

Archive proc chưa xác minh tính nhân quả; nhãn và các trạm core chưa có kiểm định thực địa. Test2023 đã được khảo sát trước; đây chỉ là đánh giá thăm dò. Phải xem MAE lúc nước cao, recall, false positives, false negatives và các lần chuyển ngưỡng, không tuyên bố hiệu quả bằng MAE tổng thấp.

Trọng số/dự đoán/biểu đồ đều giữ cục bộ. Nguồn: FloodNet (New York University and The City University of New York). Rà quyền trước chia sẻ dữ liệu dẫn xuất hoặc trọng số. Phân tích/biểu diễn phái sinh được phép: CC BY-NC-SA 4.0 cùng Data Access License Agreement.

## Chẩn đoán và thí nghiệm delta — bổ sung 10/11/2026

Kết quả tổng hợp: [KET_QUA_CHAN_DOAN_VA_DELTA.md](KET_QUA_CHAN_DOAN_VA_DELTA.md).

```powershell
# Chẩn đoán model cũ, không fit; tạo diagnostic_runs/<timestamp>/ mới.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\run_relocated.py" "D:\CODE\IOT\IOT training\analysis\diagnose_training.py"

# Fit 4 cấu hình delta trên đúng train cũ; chỉ đánh giá validation, KHÔNG nạp test.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\run_relocated.py" "D:\CODE\IOT\IOT training\analysis\delta_experiment.py"

# Kiểm tra lại lượt delta đã có, không refit hoặc nạp test.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\run_relocated.py" "D:\CODE\IOT\IOT training\analysis\validate_delta.py" "D:\CODE\IOT\IOT training\analysis\training_runs\20261110T040220_424848Z_delta"
```

`diagnose_training.py --require-improvement` là phép kiểm tra tái lập triệu chứng **của model cũ**: cố ý thất bại vì model cũ thua baseline và được giữ nguyên, không phải lỗi script. `validate_delta.py <run> --require-improvement` kiểm tra model delta tốt nhất có giảm MAE cân bằng trên validation hay không; đây là cổng thực nghiệm, không phải chứng minh an toàn hay một unit test bắt buộc mọi lượt train phải thắng.

`delta_experiment.py` tái sử dụng loader/metric từ lượt đầu, kiểm tra checksum đầu vào và runtime, chặn split test, lưu cả bốn model cùng dự đoán validation cục bộ. Chỉ đổi đích fit thành `future_depth - current_depth`; clipping về 0 diễn ra SAU khi cộng delta với độ sâu hiện tại. Không dùng ID, metadata tương lai, cờ đối soát hoặc phân tầng chẩn đoán làm feature. Không cài thêm thư viện.

## Nhánh raw riêng — bổ sung 10/11/2026

Đã train nhánh raw trên đúng ba điểm, không mở rộng bảy trạm. Raw chỉ được xử lý bằng các quan trắc đến thời điểm bin hoàn tất; proc chỉ làm nhãn tương lai/đối chứng, không làm feature. Không refit hoặc đổi model proc đang lưu.

- [Kết quả nhánh raw](KET_QUA_NHANH_RAW.md).
- [Quy tắc và lệnh chạy](raw_branch/README.md).
- Lượt đã hoàn thành: `training_runs/20261110T043429_532355Z_raw_delta`.

```powershell
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\run_relocated.py" "D:\CODE\IOT\IOT training\analysis\train_raw_delta.py"
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\run_relocated.py" "D:\CODE\IOT\IOT training\analysis\validate_raw.py" "D:\CODE\IOT\IOT training\analysis\training_runs\20261110T043429_532355Z_raw_delta"
```

Lệnh đầu tạo run mới, không nên chạy lại chỉ vì đọc tài liệu. Lệnh thứ hai dựng lại cửa sổ, đối soát checksum/membership và tái lập dự đoán từ model đã lưu, không fit và không đọc test. Chưa dùng nhánh raw để cảnh báo/định tuyến ngoài thực tế.

## Mở rộng raw từ 3 lên 10 mã triển khai — bổ sung 10/11/2026

- [Kết quả mở rộng](KET_QUA_MO_RONG_RAW.md).
- [Chính sách và lệnh chạy](raw_expansion/README.md).
- Run: `training_runs/20261110T050356_109133Z_raw_expanded`.

Cả bảy ứng viên đạt điều kiện độ phủ train, nhưng vẫn có cờ chất lượng chưa xác minh. Giữ toàn bộ cửa sổ raw hợp lệ, không dùng mask offline. Train tăng từ 270.312 lên 542.217; validation cố định 92.215 hàng. Mở rộng chưa cải thiện MAE cân bằng (0,4949→0,5402cm); phương pháp được chọn vẫn là raw delta ba mã `d3_weight20`. Không tự thay model cũ hoặc triển khai cảnh báo.

```powershell
# Kiểm tra lại lượt mở rộng đã có; không fit hoặc đọc test.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\run_relocated.py" "D:\CODE\IOT\IOT training\analysis\validate_raw_expanded.py" "D:\CODE\IOT\IOT training\analysis\training_runs\20261110T050356_109133Z_raw_expanded"
```

## Phân tích tác động mở rộng — phần A (11/11/2026)

[Kết quả và hướng dẫn](KET_QUA_PHAN_TICH_MO_RONG_RAW.md). Không train lại, không đọc test, không tự đổi model; tạo thư mục chẩn đoán mới.

```powershell
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\diagnose_raw_expansion.py"
```

## Đánh giá cảnh báo theo đợt — phần B (11/11/2026)

[Kết quả và quy tắc](KET_QUA_CANH_BAO_THEO_DOT.md). Chỉ dùng validation đã đóng băng; không fit, không đọc test hoặc tự đổi model/cảnh báo. Tạo run mới, giữ nguyên artifacts cũ.

```powershell
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\evaluate_warning_events.py"

# Validator chỉ đọc, không ghi báo cáo lại.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\validate_warning_events.py" "D:\CODE\IOT\IOT training\analysis\diagnostic_runs\20261111T004533_510057Z_warning_events"
```

## Bản đồ dự báo và thí nghiệm độ sâu (11/11/2026)

[Kết quả / cách mở bản đồ / lệnh chạy](KET_QUA_BAN_DO_DU_BAO_DO_SAU.md).

- `export_forecast_replay.py`: chỉ export model raw ba mã cũ ra thư mục cục bộ gitignore; chọn file trên frontend `/forecast`, không đưa vào public.
- `depth_accuracy.py`: một thử nghiệm riêng đổi objective sang absolute error; không quét cấu hình, không đọc test, không tự thay model.
- `validate_depth_accuracy.py <run>`: tái lập bốn phương pháp và các chỉ số/nhóm lỗi, chỉ đọc.
- MAE toàn tập của thử nghiệm giảm nhưng MAE nhãn cao xấu hơn, MAE cân bằng gần như hòa nhưng hơi kém. Giữ model cũ cho demo; chưa có kiểm định độc lập cho model mới.
