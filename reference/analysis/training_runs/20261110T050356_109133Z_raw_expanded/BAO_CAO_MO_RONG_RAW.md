# Mở rộng điểm train cho nhánh raw

**Thử nghiệm hồi cứu, không xác nhận nhãn sạch hoặc phê duyệt cảnh báo/định tuyến.**

## Thiết kế

- Giữ nguyên nhánh raw ba điểm, preprocessing 5 phút, nhãn proc tương lai, 12 feature, bốn cấu hình XGBoost delta và mốc chia tập/embargo.
- Chọn điểm bổ sung CHỈ theo train: ≥1.000 cửa sổ, ≥30 ngày, ≥20 nhãn cao, ≥1.000 nhãn thấp, ≥3 ngày và ≥3 nhóm nhãn cao cách nhau hơn 6 giờ.
- Nhóm 6 giờ không phải các trận mưa độc lập. Đạt độ phủ không đồng nghĩa ngập thật hoặc nhãn sạch.
- Đọc cache ứng viên trước 01/01/2023, không đọc validation của điểm mới hoặc test. Cảnh báo chất lượng không bị giấu.
- Với điểm đạt độ phủ, giữ TẤT CẢ cửa sổ raw hợp lệ, kể cả ca cao/nhảy mạnh/nghi vật cản; không dùng mask offline hoặc chỉ lấy ca đẹp.
- Đây là thử nghiệm bổ sung toàn bộ cửa sổ đủ điều kiện, KHÔNG nâng bảy điểm thành core đã xác minh sạch.

Số hàng: **{'core_train': 270312, 'added_train': 271905, 'expanded_train': 542217, 'validation': 92215, 'core_train_wet': 1220, 'expanded_train_wet': 7789}**. Bổ sung **7** điểm, tổng **10** mã triển khai train (`deployment_id`), không mặc định là từng ấy vị trí độc lập.

## Kiểm kê ứng viên — chỉ train

| Điểm | Cửa sổ | Nhãn ≥10cm | Ngày nhãn cao | Nhóm 6h | Tỷ lệ lịch sử raw âm | Chọn | Lý do không đạt |
|---|---:|---:|---:|---:|---:|---|---|
| daily_gentle_beetle | 46075 | 136 | 8 | 7 | 62.7% | True |  |
| evenly_divine_dingo | 19354 | 484 | 5 | 3 | 52.6% | True |  |
| overly_heroic_squid | 46080 | 322 | 7 | 5 | 83.0% | True |  |
| simply_pet_joey | 20549 | 2430 | 16 | 8 | 10.0% | True |  |
| tired_red_monkey | 20991 | 1608 | 42 | 40 | 74.6% | True |  |
| weekly_fair_dove | 52606 | 1322 | 36 | 26 | 72.3% | True |  |
| weekly_poetic_guinea | 66250 | 267 | 7 | 7 | 57.3% | True |  |

Raw âm đã quy về 0 theo chính sách nhánh raw cũ, không tự kết luận là lỗi. Các điểm này từng có dấu hiệu mức cao kéo dài, lệch mốc hoặc dao động mạnh; `station_audit_train.csv` ghi thêm sparse/mismatch và các cờ, không loại cửa sổ chỉ vì cờ.

## Validation cố định — mọi phương pháp trên đúng cùng hàng/nhãn

| Model | MAE toàn tập | MAE ≥10cm | MAE cân bằng | FP | FN |
|---|---:|---:|---:|---:|---:|
| persistence | 0.1212 | 0.9500 | 0.5309 | 70 | 83 |
| linear_trend_5min | 0.1619 | 0.9944 | 0.5734 | 192 | 75 |
| three_d3_uniform | 0.0799 | 0.9839 | 0.5268 | 64 | 89 |
| three_d3_weight20 | 0.1029 | 0.8960 | 0.4949 | 246 | 57 |
| three_d4_uniform | 0.0794 | 0.9428 | 0.5062 | 70 | 85 |
| three_d4_weight20 | 0.1012 | 0.9091 | 0.5006 | 268 | 59 |
| expanded_d3_uniform | 0.1399 | 3.9245 | 2.0107 | 10 | 550 |
| expanded_d3_weight20 | 0.1357 | 0.9539 | 0.5402 | 133 | 70 |
| expanded_d4_uniform | 0.1213 | 3.1512 | 1.6190 | 15 | 299 |
| expanded_d4_weight20 | 0.1536 | 0.9875 | 0.5658 | 171 | 72 |

Ba điểm tốt nhất: **three_d3_weight20**. Nhóm mở rộng tốt nhất: **expanded_d3_weight20**.
Phương pháp được chọn trong cả baseline/hai nhóm: **three_d3_weight20**.
Chênh lệch MAE cân bằng expanded − three: **+0.045239 cm** (âm tốt hơn).

Không ép kết luận thêm dữ liệu là tốt hơn. Không thay tiêu chí sau khi xem kết quả; FP/FN vẫn phải đọc riêng.

## So từng cấu hình giống nhau

| Cấu hình | MAE cân bằng 3 điểm | Mở rộng | Thay đổi cm | FP 3 / mở rộng | FN 3 / mở rộng |
|---|---:|---:|---:|---:|---:|
| d3_uniform | 0.5268 | 2.0107 | +1.4839 | 64 / 10 | 89 / 550 |
| d3_weight20 | 0.4949 | 0.5402 | +0.0452 | 246 / 133 | 57 / 70 |
| d4_uniform | 0.5062 | 1.6190 | +1.1129 | 70 / 15 | 85 / 299 |
| d4_weight20 | 0.5006 | 0.5658 | +0.0652 | 268 / 171 | 59 / 72 |

## Giới hạn

- Validation vẫn chỉ có hai trạm core cũ, không chứng minh dự báo tốt ở các trạm mới.
- Validation được dùng nhiều lần; không là bộ xác nhận độc lập. Không nạp test để điều chỉnh.
- Lịch sử raw chỉ dùng quá khứ theo timestamp archive, chưa tái dựng độ trễ nhận hoặc hiệu chuẩn nguồn.
- Nhãn proc tương lai là tham chiếu hồi cứu chưa xác nhận hiện trường; không phải dự báo đúng tức thời t+10.
- Các cửa sổ tương quan; FP/FN không phải số đợt cảnh báo độc lập.
- Không sửa cảnh báo theo ngưỡng, backend/frontend hoặc dữ liệu/artifacts cũ.

## Artifacts

- `config_before_preparation.json`, `station_selection_before_fit.json`, `config_before_fit.json`: chính sách/quyết định trước fit.
- `station_audit_train.csv`, `candidate_exclusion_reasons.csv`, `candidate_train/`: kiểm kê và tất cả cửa sổ ứng viên, không giấu ca khó.
- `frozen_input_provenance.json`, `source_provenance.json`, `candidate_window_provenance.json`: nguồn/checksum.
- `expanded_*.ubj`, `metrics.csv`, `selection_validation_only.json`, `validation_predictions_local.parquet`: bốn model mới và điểm trên validation cố định.
- `matched_config_comparison.csv`, `validation_breakdown.csv`: so từng cấu hình và từng trạm.

Nguồn: FloodNet (New York University and The City University of New York). Giữ artifacts cục bộ; không phân phối dữ liệu/trọng số khi chưa làm rõ quyền. Phân tích/biểu diễn phái sinh được phép theo CC BY-NC-SA 4.0 cùng Data Access License Agreement.
