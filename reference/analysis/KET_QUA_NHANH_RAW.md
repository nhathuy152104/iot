# Kết quả nhánh raw trên ba điểm

**Ngày:** 10/11/2026. **Run:** `training_runs/20261110T043429_532355Z_raw_delta`.

**Kết luận:** đã xây và train nhánh raw riêng. XGBoost delta giảm MAE cân bằng so với baseline dùng cùng raw, nhưng báo giả vẫn nhiều hơn. Nhánh raw chưa tốt hơn nhánh proc về MAE cân bằng. Chưa triển khai cảnh báo/định tuyến/ESP32.

## 1. Đã giữ và đã thay đổi gì?

- Giữ đúng ba trạm, mốc chia train/validation, embargo 24h, bước 5 phút, 12 feature và bốn cấu hình XGBoost delta. Không thêm bảy trạm, không đọc test.
- Thay đầu vào: trung vị raw của từng bin 5 phút đã hoàn tất, raw âm quy về 0 và có cờ thống kê; không nhìn mẫu sau thời điểm quyết định.
- Bin rỗng hoặc raw ở hàng cuối không hữu hạn thì thiếu; không fill/nội suy. Không tự loại mức cao hoặc nước nhảy mạnh.
- **Nhãn tương lai vẫn là proc** để giữ bài toán đối chiếu. Proc không vào feature hoặc điều kiện hợp lệ của lịch sử raw.
- Trung vị không thể nhận diện mọi vật cản/nhiễu, đặc biệt ở bin một mẫu. Không có dữ liệu thời điểm nhận bản tin để chứng minh độ trễ vận hành thật.

## 2. Số cửa sổ — không bỏ ca khó ngầm

| Tập | Proc trước | Raw mới | Chung | Chỉ proc | Chỉ raw |
|---|---:|---:|---:|---:|---:|
| Train | 270.313 | 270.312 | 270.312 | 1 | 0 |
| Validation | 92.215 | 92.215 | 92.215 | 0 | 0 |

Một cửa sổ train của `widely_proud_lizard` tại **10/12/2021 20:15 UTC** không có nhãn tương lai theo quy tắc bin hoàn tất ở cuối chuỗi nguồn của trạm; nhánh proc cũ có dùng bin cuối đó. Đây là mẫu có nhãn cũ <10 cm. Số nhãn ≥10 cm không đổi: **1.220 train, 1.037 validation**. Không loại vì model sai hoặc vì mức nước cao.

Mọi cửa sổ mất/thêm và lý do có trong `window_coverage.csv`, `exclusion_reasons.csv` và `window_membership_local.parquet`. Raw âm chỉ được biến đổi cho feature và có thống kê trong `raw_bin_audit.csv`; CSV gốc không thay đổi.

## 3. Kết quả trên cùng 92.215 cửa sổ validation

| Phương pháp | MAE toàn tập cm | MAE nhãn ≥10cm | MAE cân bằng cm | Báo giả | Bỏ sót |
|---|---:|---:|---:|---:|---:|
| Raw persistence | 0,1212 | 0,9500 | 0,5309 | **70** | 83 |
| Raw xu hướng tuyến tính | 0,1619 | 0,9944 | 0,5734 | 192 | 75 |
| **Raw delta d3_weight20** | **0,1029** | **0,8960** | **0,4949** | 246 | **57** |
| Proc persistence — đối chứng cũ | 0,0693 | 0,9104 | 0,4851 | 78 | 79 |
| Proc delta d3_weight20 — đối chứng cũ | 0,1083 | 0,8450 | 0,4725 | 324 | 62 |

`raw_delta_d3_weight20` được chọn bằng MAE cân bằng trên validation raw. So với raw persistence, MAE cân bằng giảm khoảng **6,8%**, bỏ sót giảm **83→57**, nhưng báo giả tăng **70→246**. Không đổi tiêu chí để chọn cấu hình đẹp hơn ở chỉ số khác.

So với proc delta, raw delta có ít cửa sổ báo giả/bỏ sót hơn trong lượt này, nhưng MAE cân bằng kém hơn (**0,4949 so với 0,4725 cm**). Không kết luận raw thắng toàn diện. Hai pipeline có lịch sử/đặc trưng khác nhau và train lệch một cửa sổ; validation trùng nhau và nhãn tương lai khớp tuyệt đối.

## 4. Kiểm tra kỹ thuật

- 14 kiểm thử mới: trung vị, raw âm, null hàng cuối, thiếu bin, chưa kết thúc bin, prefix replay, thay đổi tương lai không đổi feature quá khứ, proc không ảnh hưởng feature, nhãn bin chính xác, phân loại lý do loại, chặn tháng test và xuất báo cáo. Tổng **49 kiểm thử đạt**: 33 ở `analysis/` và 16 pipeline/đối soát 5 phút, không trùng nhau.
- Dựng lại toàn bộ 270.312 cửa sổ train và 92.215 validation từ 51 cache có checksum, xác minh 33 fingerprint CSV nguồn, 6 file cửa sổ raw và 5 file cửa sổ proc: **PASS**.
- Prefix replay trên cả ba trạm ở mốc train: feature không đổi so với xử lý toàn chuỗi.
- Nạp lại bốn model raw và hai baseline; tái lập mọi dự đoán/chỉ số validation và quyết định chọn model. Đối chiếu thêm năm phương pháp raw/proc trên giao validation: **PASS**.
- Không refit khi kiểm tra, không nạp test, không chỉnh backend/frontend hoặc artifacts proc.

Các kiểm tra này không xác nhận nhãn ngập thật, hiệu chuẩn nguồn, độ trễ mạng, độ an toàn hoặc khả năng dùng model NYC ở địa bàn ESP32.

## 5. Đọc chi tiết và chạy lại

- [Báo cáo đầy đủ](training_runs/20261110T043429_532355Z_raw_delta/BAO_CAO_RAW.md).
- [Hướng dẫn nhánh raw](raw_branch/README.md).
- Model ứng viên: `training_runs/20261110T043429_532355Z_raw_delta/raw_delta_d3_weight20.ubj`; cần đi cùng đúng chính sách preprocessing, không thay thế model proc trực tiếp.
- Kết quả kiểm tra: `training_runs/20261110T043429_532355Z_raw_delta/artifact_validation.json`.

**Tiếp theo:** nếu mở rộng điểm, làm lượt riêng giữ nguyên nhánh raw hiện tại làm đối chứng; đồng thời đánh giá báo giả/bỏ sót theo sự kiện trước khi tích hợp cảnh báo. Không cần chờ FloodNet xác nhận mới tiếp tục nghiên cứu, nhưng không tuyên bố đã chứng minh hiệu quả thời gian thực.

**Cập nhật sau lượt này:** đã thực hiện [thí nghiệm mở rộng lên 10 mã triển khai](KET_QUA_MO_RONG_RAW.md). Chưa cải thiện MAE cân bằng trên validation cố định; model ba mã vẫn được giữ làm ứng viên đối chứng.

Nguồn: FloodNet (New York University and The City University of New York). Giữ artifacts cục bộ; làm rõ quyền trước phân phối dữ liệu hoặc trọng số. Phân tích/biểu diễn phái sinh được phép theo CC BY-NC-SA 4.0 cùng Data Access License Agreement.
