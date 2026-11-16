# Nhánh raw trên ba điểm: baseline và XGBoost delta

**Đầu vào raw được xử lý chỉ theo quá khứ trên đồng hồ archive. Nhãn tương lai vẫn là proc hồi cứu. Chưa chứng minh hiệu quả vận hành thời gian thực hoặc chuyển sang ESP32.**

## 1. Quy tắc chốt trước khi chuẩn bị/train

- Ba điểm giữ nguyên: daily_new_falcon, ugliest_cyan_elephant, widely_proud_lizard. Không thêm bảy trạm hoặc xếp hạng lại.
- Bin UTC (t−5,t]: trung vị raw hữu hạn, sau khi quy raw âm về 0. Chỉ hoàn tất bin tại t; không dùng mẫu sau t.
- Bin rỗng hoặc raw ở hàng vật lý cuối là null/không hữu hạn: đánh dấu thiếu, không thay bằng số đo cũ.
- Không nội suy/fill, không cắt trên, không loại nước cao/nhảy mạnh hoặc dùng mask offline. Ghi rõ số raw âm và mẫu cao còn giữ.
- Trung vị giảm một phần xung nhiễu khi bin có nhiều mẫu; bin chỉ một mẫu và vật cản kéo dài vẫn có thể gây sai.
- 12 feature có ý nghĩa như trước nhưng được tính từ raw-derived depth. Proc KHÔNG vào feature hoặc điều kiện lịch sử raw.
- Nhãn vẫn là proc cuối bin (t+5,t+10], chỉ dùng làm giám sát tương lai. Nhãn không phải cực đại hoặc đúng tức thời t+10.
- Giữ mốc 01/01/2023, 01/07/2023, embargo 24h và kiểm tra support đầy đủ 15 phút lịch sử/10 phút tương lai.
- Không đọc test hoặc cache nguồn từ 07/2023 trở đi. Chỉ chọn bằng validation raw-eligible.

## 2. Số cửa sổ thay đổi

| Tập | Trạm | Proc cũ | Raw mới | Chung | Chỉ raw | Chỉ proc | Nhãn ≥10cm raw / proc |
|---|---|---:|---:|---:|---:|---:|---:|
| train | daily_new_falcon | 208388 | 208388 | 208388 | 0 | 0 | 737 / 737 |
| validation | daily_new_falcon | 51444 | 51444 | 51444 | 0 | 0 | 8 / 8 |
| train | ugliest_cyan_elephant | 33849 | 33849 | 33849 | 0 | 0 | 427 / 427 |
| validation | ugliest_cyan_elephant | 40771 | 40771 | 40771 | 0 | 0 | 1029 / 1029 |
| train | widely_proud_lizard | 28076 | 28075 | 28075 | 0 | 1 | 56 / 56 |
| validation | widely_proud_lizard | 0 | 0 | 0 | 0 | 0 | 0 / 0 |

Tổng cửa sổ raw: **{'train': 270312, 'validation': 92215}**.

Các cửa sổ cũ bị mất, gồm cả nhãn cao nếu có, được giữ dấu vết trong `window_coverage.csv` và `window_membership_local.parquet`. Không bỏ vì model dự báo sai.

### Lý do loại, phân nhóm không chồng nhau

| Tập | Trạm | Lý do | Số bin quyết định |
|---|---|---|---:|
| train | daily_new_falcon | kept | 208388 |
| train | daily_new_falcon | missing_or_invalid_current_raw_bin | 19322 |
| train | daily_new_falcon | missing_or_invalid_lag5_raw_bin | 2987 |
| train | daily_new_falcon | missing_or_invalid_lag10_raw_bin | 2399 |
| train | daily_new_falcon | missing_or_invalid_future_proc_label | 2149 |
| validation | daily_new_falcon | kept | 51444 |
| validation | daily_new_falcon | missing_or_invalid_current_raw_bin | 80 |
| validation | daily_new_falcon | missing_or_invalid_future_proc_label | 9 |
| validation | daily_new_falcon | missing_or_invalid_lag5_raw_bin | 7 |
| validation | daily_new_falcon | missing_or_invalid_lag10_raw_bin | 7 |
| train | ugliest_cyan_elephant | kept | 33849 |
| train | ugliest_cyan_elephant | missing_or_invalid_current_raw_bin | 9855 |
| train | ugliest_cyan_elephant | missing_or_invalid_future_proc_label | 245 |
| train | ugliest_cyan_elephant | missing_or_invalid_lag5_raw_bin | 243 |
| train | ugliest_cyan_elephant | missing_or_invalid_lag10_raw_bin | 242 |
| validation | ugliest_cyan_elephant | kept | 40771 |
| validation | ugliest_cyan_elephant | missing_or_invalid_current_raw_bin | 10267 |
| validation | ugliest_cyan_elephant | missing_or_invalid_future_proc_label | 177 |
| validation | ugliest_cyan_elephant | missing_or_invalid_lag5_raw_bin | 166 |
| validation | ugliest_cyan_elephant | missing_or_invalid_lag10_raw_bin | 166 |
| train | widely_proud_lizard | kept | 28075 |
| train | widely_proud_lizard | missing_or_invalid_current_raw_bin | 6068 |
| train | widely_proud_lizard | missing_or_invalid_lag5_raw_bin | 3161 |
| train | widely_proud_lizard | missing_or_invalid_lag10_raw_bin | 2722 |
| train | widely_proud_lizard | missing_or_invalid_future_proc_label | 2506 |

Ưu tiên lý do theo thứ tự: raw bin hiện tại thiếu → lag5 thiếu → lag10 thiếu → nhãn tương lai thiếu/âm → feature không hữu hạn → kept. Tổng các lý do bằng tổng bin trong support từng trạm/tập. Bin missing bao gồm khoảng không quan sát và null; xem tách riêng ở `raw_bin_audit.csv`.

## 3. Validation chính: toàn bộ cửa sổ raw đủ điều kiện

| Model | Hàng | Nhãn ≥10cm | MAE toàn tập | MAE ≥10cm | MAE cân bằng | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| persistence | 92215 | 1037 | 0.1212 | 0.9500 | 0.5309 | 70 | 83 |
| linear_trend_5min | 92215 | 1037 | 0.1619 | 0.9944 | 0.5734 | 192 | 75 |
| raw_delta_d3_uniform | 92215 | 1037 | 0.0799 | 0.9839 | 0.5268 | 64 | 89 |
| raw_delta_d3_weight20 | 92215 | 1037 | 0.1029 | 0.8960 | 0.4949 | 246 | 57 |
| raw_delta_d4_uniform | 92215 | 1037 | 0.0794 | 0.9428 | 0.5062 | 70 | 85 |
| raw_delta_d4_weight20 | 92215 | 1037 | 0.1012 | 0.9091 | 0.5006 | 268 | 59 |

Delta tốt nhất: **raw_delta_d3_weight20**. Phương pháp được chọn trong cả baseline/AI: **raw_delta_d3_weight20**. Không ép AI thắng.

## 4. Đối chiếu raw/proc trên đúng cùng cửa sổ validation

| Model | Hàng | Nhãn ≥10cm | MAE toàn tập | MAE ≥10cm | MAE cân bằng | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_persistence | 92215 | 1037 | 0.1212 | 0.9500 | 0.5309 | 70 | 83 |
| raw_linear_trend_5min | 92215 | 1037 | 0.1619 | 0.9944 | 0.5734 | 192 | 75 |
| raw_delta_selected | 92215 | 1037 | 0.1029 | 0.8960 | 0.4949 | 246 | 57 |
| proc_persistence | 92215 | 1037 | 0.0693 | 0.9104 | 0.4851 | 78 | 79 |
| proc_delta_selected | 92215 | 1037 | 0.1083 | 0.8450 | 0.4725 | 324 | 62 |

Các model ở bảng này được chốt TRƯỚC khi tính bảng đối chiếu. Nhãn tương lai ở các hàng chung khớp tuyệt đối. Chỉ đối chiếu điểm raw/proc trực tiếp trên giao cửa sổ này; kiểm tra bảng coverage trước khi coi toàn validation hai nhánh là cùng mẫu số. Lượt này: 92,215 hàng chung; raw 92,215, proc 92,215. Toàn bộ validation trùng nhau.

**Giới hạn so sánh:** lịch sử raw/proc và số hàng train có thể khác. Giao validation loại khác biệt mẫu số đánh giá, nhưng không biến đây thành thí nghiệm chỉ thay đúng một biến trong toàn bộ quá trình học. `new_crossing` nếu xem trong CSV là tương đối với mức hiện tại của từng pipeline, không phải cùng nhãn khởi phát độc lập.

## 5. Kiểm tra và giới hạn

- Cả bốn model lưu/nạp lại khớp mọi dự đoán validation. Kiểm tra độc lập ở `validate_raw.py`.
- Prefix replay trên train kiểm tra feature cùng thời điểm không thay đổi khi có thêm dữ liệu tương lai.
- CSV chỉ có timestamp archive, không có thời điểm nhận raw. Chưa mô phỏng độ trễ mạng hay xác nhận hiệu chuẩn nguồn.
- Nhãn proc có thể qua xử lý dùng mẫu sau. Dùng làm nhãn hồi cứu không đưa nó vào đầu vào, nhưng không xác minh nhãn đúng.
- Việc chọn ba trạm ban đầu dựa trên train proc; không tuyên bố việc chọn trạm hoàn toàn độc lập proc.
- Raw âm quy về 0 chỉ là quy ước đầu vào, không sửa mốc hiệu chuẩn hoặc chứng minh đường khô.
- Các cửa sổ tương quan, validation được xem nhiều lần; cần đánh giá theo sự kiện và kiểm định độc lập.
- Không sửa backend/frontend, không đưa model vào cảnh báo/định tuyến hoặc thay thế luật ngưỡng.

## 6. Artifacts

- `config_before_preparation.json`, `config_before_fit.json`: chính sách/cấu hình chốt trước thực nghiệm.
- `source_provenance.json`, `proc_window_provenance.json`, `window_provenance.json`: nguồn và checksum.
- `windows/`, `window_membership_local.parquet`: cửa sổ và dấu vết khác biệt với nhánh proc, giữ cục bộ.
- `raw_delta_*.ubj`, `selection_validation_only.json`, `metrics.csv`, `summary.json`: model và lựa chọn.
- `paired_validation_metrics.csv`, `validation_breakdown.csv`: đối chiếu/phân tầng, không lọc ca khó.

Nguồn: FloodNet (New York University and The City University of New York). Giữ mọi dữ liệu/đầu ra/trọng số cục bộ; làm rõ quyền trước phân phối. Phân tích/biểu diễn phái sinh được phép theo CC BY-NC-SA 4.0 cùng Data Access License Agreement.
