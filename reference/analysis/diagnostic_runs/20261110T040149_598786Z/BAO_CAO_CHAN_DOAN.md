# Chẩn đoán lượt train đầu tiên

**Không fit lại, không chọn model, không loại hàng. Test cũ chỉ dùng chẩn đoán hồi cứu.**

PASS: all original validation/test metrics reproduced from saved model and checksummed inputs

Khoảng thua baseline trên validation: **+0.027922 cm MAE cân bằng**.

## 1. Phạm vi độ sâu

| Tập | Trạm | Số nhãn ≥10cm | Nhãn lớn nhất cm |
|---|---|---:|---:|
| train | ALL_CORE | 1220 | 97.80 |
| train | daily_new_falcon | 737 | 34.00 |
| train | ugliest_cyan_elephant | 427 | 62.80 |
| train | widely_proud_lizard | 56 | 97.80 |
| validation | ALL_CORE | 1037 | 33.30 |
| validation | daily_new_falcon | 8 | 11.70 |
| validation | ugliest_cyan_elephant | 1029 | 33.30 |
| test | ALL_CORE | 137 | 118.60 |
| test | daily_new_falcon | 6 | 13.30 |
| test | ugliest_cyan_elephant | 131 | 118.60 |

Nhãn vượt cực đại train là bằng chứng thiếu độ phủ, không chứng minh nhãn sai. Cây hồi quy dự báo độ sâu tuyệt đối khó ngoại suy mức chưa gặp; kiểm tra bảng support bên dưới thay vì mặc định đó là nguyên nhân duy nhất.

## 2. Phân rã sai số model cũ

| Tập | Nhóm | Model | Hàng | Nhãn ≥10 | MAE ≥10cm | Tổng lỗi nhãn ≥10cm | FP | FN |
|---|---|---|---:|---:|---:|---:|---:|---:|
| validation | all | persistence | 92215 | 1037 | 0.910 | 944.10 | 78 | 79 |
| validation | all | linear_trend_5min | 92215 | 1037 | 1.159 | 1201.70 | 296 | 73 |
| validation | all | original_absolute_selected | 92215 | 1037 | 0.899 | 932.72 | 336 | 59 |
| validation | future_le_pooled_train_max | persistence | 92215 | 1037 | 0.910 | 944.10 | 78 | 79 |
| validation | future_le_pooled_train_max | linear_trend_5min | 92215 | 1037 | 1.159 | 1201.70 | 296 | 73 |
| validation | future_le_pooled_train_max | original_absolute_selected | 92215 | 1037 | 0.899 | 932.72 | 336 | 59 |
| validation | down_crossing | persistence | 78 | 0 | N/A | 0.00 | 78 | 0 |
| validation | down_crossing | linear_trend_5min | 78 | 0 | N/A | 0.00 | 58 | 0 |
| validation | down_crossing | original_absolute_selected | 78 | 0 | N/A | 0.00 | 75 | 0 |
| validation | falling_gt_1cm | persistence | 774 | 54 | 1.683 | 90.90 | 0 | 0 |
| validation | falling_gt_1cm | linear_trend_5min | 774 | 54 | 1.476 | 79.70 | 113 | 0 |
| validation | falling_gt_1cm | original_absolute_selected | 774 | 54 | 1.464 | 79.05 | 109 | 0 |
| validation | new_crossing | persistence | 79 | 79 | 5.268 | 416.20 | 0 | 79 |
| validation | new_crossing | linear_trend_5min | 79 | 79 | 5.537 | 437.40 | 0 | 58 |
| validation | new_crossing | original_absolute_selected | 79 | 79 | 4.262 | 336.70 | 0 | 59 |
| validation | rising_gt_1cm | persistence | 847 | 71 | 2.154 | 152.90 | 0 | 0 |
| validation | rising_gt_1cm | linear_trend_5min | 847 | 71 | 2.131 | 151.30 | 19 | 2 |
| validation | rising_gt_1cm | original_absolute_selected | 847 | 71 | 2.108 | 149.64 | 22 | 0 |
| validation | stable_high | persistence | 833 | 833 | 0.341 | 284.10 | 0 | 0 |
| validation | stable_high | linear_trend_5min | 833 | 833 | 0.640 | 533.30 | 0 | 13 |
| validation | stable_high | original_absolute_selected | 833 | 833 | 0.441 | 367.33 | 0 | 0 |
| validation | stable_low | persistence | 89604 | 0 | N/A | 0.00 | 0 | 0 |
| validation | stable_low | linear_trend_5min | 89604 | 0 | N/A | 0.00 | 106 | 0 |
| validation | stable_low | original_absolute_selected | 89604 | 0 | N/A | 0.00 | 130 | 0 |
| test | all | persistence | 42218 | 137 | 5.507 | 754.50 | 43 | 43 |
| test | all | linear_trend_5min | 42218 | 137 | 9.066 | 1242.10 | 132 | 40 |
| test | all | original_absolute_selected | 42218 | 137 | 14.838 | 2032.77 | 154 | 33 |
| test | future_gt_pooled_train_max | persistence | 27 | 27 | 5.800 | 156.60 | 0 | 1 |
| test | future_gt_pooled_train_max | linear_trend_5min | 27 | 27 | 18.696 | 504.80 | 0 | 1 |
| test | future_gt_pooled_train_max | original_absolute_selected | 27 | 27 | 54.776 | 1478.95 | 0 | 1 |
| test | future_le_pooled_train_max | persistence | 42191 | 110 | 5.435 | 597.90 | 43 | 42 |
| test | future_le_pooled_train_max | linear_trend_5min | 42191 | 110 | 6.703 | 737.30 | 132 | 39 |
| test | future_le_pooled_train_max | original_absolute_selected | 42191 | 110 | 5.035 | 553.82 | 154 | 32 |
| test | future_gt_own_station_train_max | persistence | 28 | 28 | 8.979 | 251.40 | 0 | 2 |
| test | future_gt_own_station_train_max | linear_trend_5min | 28 | 28 | 21.414 | 599.60 | 0 | 2 |
| test | future_gt_own_station_train_max | original_absolute_selected | 28 | 28 | 56.205 | 1573.73 | 0 | 2 |
| test | down_crossing | persistence | 43 | 0 | N/A | 0.00 | 43 | 0 |
| test | down_crossing | linear_trend_5min | 43 | 0 | N/A | 0.00 | 35 | 0 |
| test | down_crossing | original_absolute_selected | 43 | 0 | N/A | 0.00 | 42 | 0 |
| test | falling_gt_1cm | persistence | 488 | 26 | 1.796 | 46.70 | 0 | 0 |
| test | falling_gt_1cm | linear_trend_5min | 488 | 26 | 5.573 | 144.90 | 47 | 0 |
| test | falling_gt_1cm | original_absolute_selected | 488 | 26 | 6.117 | 159.04 | 48 | 0 |
| test | new_crossing | persistence | 43 | 43 | 14.442 | 621.00 | 0 | 43 |
| test | new_crossing | linear_trend_5min | 43 | 43 | 14.000 | 602.00 | 0 | 34 |
| test | new_crossing | original_absolute_selected | 43 | 43 | 13.452 | 578.44 | 0 | 33 |
| test | rising_gt_1cm | persistence | 454 | 14 | 4.779 | 66.90 | 0 | 0 |
| test | rising_gt_1cm | linear_trend_5min | 454 | 14 | 9.164 | 128.30 | 17 | 0 |
| test | rising_gt_1cm | original_absolute_selected | 454 | 14 | 14.451 | 202.32 | 16 | 0 |
| test | stable_high | persistence | 54 | 54 | 0.369 | 19.90 | 0 | 0 |
| test | stable_high | linear_trend_5min | 54 | 54 | 6.794 | 366.90 | 0 | 6 |
| test | stable_high | original_absolute_selected | 54 | 54 | 20.240 | 1092.97 | 0 | 0 |
| test | stable_low | persistence | 41136 | 0 | N/A | 0.00 | 0 | 0 |
| test | stable_low | linear_trend_5min | 41136 | 0 | N/A | 0.00 | 33 | 0 |
| test | stable_low | original_absolute_selected | 41136 | 0 | N/A | 0.00 | 48 | 0 |

Các pha dùng chênh lệch NHÃN TƯƠNG LAI để phân tích: chuyển ngưỡng lên/xuống trước; sau đó tăng/giảm >1cm; còn lại giữ mức cao/thấp. Chỉ là phân tầng hồi cứu, không phải feature hay bộ lọc online.

## 3. Tác động của trọng số — chỉ đối chiếu validation cũ

| Cấu hình cũ | MAE cân bằng | MAE <10cm | FP | FN |
|---|---:|---:|---:|---:|
| d3_uniform | 0.5824 | 0.0732 | 47 | 98 |
| d3_weight20 | 0.5130 | 0.1265 | 336 | 59 |
| d4_uniform | 0.5790 | 0.0727 | 58 | 98 |
| d4_weight20 | 0.5208 | 0.1134 | 525 | 54 |

Đối chiếu từng cặp cùng depth/số cây để thấy đánh đổi của weight20. Không điều chỉnh trọng số theo test.

## 4. Các đoạn đóng góp lỗi lớn nhất

Nhóm các cửa sổ hoạt động cùng trạm, cách nhau không quá 30 phút. Đây KHÔNG phải trận mưa/ngập độc lập; số hàng là cửa sổ hoạt động, không phải toàn bộ độ phủ thời gian.

| Tập | Đoạn | Hàng | Tổng lỗi XGB cm | Tổng lỗi persistence cm |
|---|---|---:|---:|---:|
| validation | validation_ugliest_cyan_elephant_20230601T1415 | 358 | 184.95 | 134.60 |
| validation | validation_ugliest_cyan_elephant_20230627T1535 | 22 | 142.16 | 116.10 |
| validation | validation_ugliest_cyan_elephant_20230429T0525 | 220 | 122.65 | 104.60 |
| validation | validation_ugliest_cyan_elephant_20230126T0000 | 124 | 87.64 | 86.80 |
| validation | validation_ugliest_cyan_elephant_20230523T1205 | 4 | 84.53 | 53.20 |
| test | test_ugliest_cyan_elephant_20230823T1210 | 32 | 1748.61 | 488.20 |
| test | test_ugliest_cyan_elephant_20230818T0950 | 6 | 238.34 | 162.20 |
| test | test_ugliest_cyan_elephant_20230818T0800 | 11 | 160.94 | 122.50 |
| test | test_ugliest_cyan_elephant_20230810T2030 | 44 | 90.88 | 87.60 |
| test | test_ugliest_cyan_elephant_20230815T1430 | 9 | 64.07 | 62.80 |

## 5. Biểu đồ nguồn và dự đoán

- [validation_ugliest_cyan_elephant_20230601T1415](validation_ugliest_cyan_elephant_20230601T1415.png): 1719 hàng nguồn; 0 hàng raw/proc lệch ≥10cm. Chưa xác minh ngập thật.
- [validation_ugliest_cyan_elephant_20230627T1535](validation_ugliest_cyan_elephant_20230627T1535.png): 199 hàng nguồn; 0 hàng raw/proc lệch ≥10cm. Chưa xác minh ngập thật.
- [validation_ugliest_cyan_elephant_20230429T0525](validation_ugliest_cyan_elephant_20230429T0525.png): 399 hàng nguồn; 0 hàng raw/proc lệch ≥10cm. Chưa xác minh ngập thật.
- [test_ugliest_cyan_elephant_20230823T1210](test_ugliest_cyan_elephant_20230823T1210.png): 169 hàng nguồn; 33 hàng raw/proc lệch ≥10cm. Chưa xác minh ngập thật.
- [test_ugliest_cyan_elephant_20230818T0950](test_ugliest_cyan_elephant_20230818T0950.png): 88 hàng nguồn; 78 hàng raw/proc lệch ≥10cm. Chưa xác minh ngập thật.
- [test_ugliest_cyan_elephant_20230818T0800](test_ugliest_cyan_elephant_20230818T0800.png): 120 hàng nguồn; 43 hàng raw/proc lệch ≥10cm. Chưa xác minh ngập thật.

Không suy ra nhãn đúng chỉ vì raw/filt/proc trùng nhau. Không có dữ liệu hiện trường/thời tiết độc lập trong phân tích này. Không loại ca cực trị để làm đẹp điểm.

## 6. Hành động có kiểm soát

- Thử delta target trên cùng train/validation; giữ nguyên 4 cấu hình, feature, trọng số và quy tắc chọn.
- Không dùng các phân tầng tương lai, nhóm đoạn hoặc biểu đồ test làm feature/bộ lọc dữ liệu.
- Xác minh causality/độ trễ của bộ lọc FloodNet trước tuyên bố hiệu quả thời gian thực.
- Bổ sung trạm/mưa/lịch sử dài hơn phải là thí nghiệm khác. Cần tập xác nhận độc lập trước khẳng định tổng quát hóa.

Nguồn: FloodNet (New York University and The City University of New York). Giữ toàn bộ artifacts cục bộ; không phân phối nguồn/trích xuất/trọng số khi chưa làm rõ quyền. Phân tích/biểu diễn phái sinh được phép: CC BY-NC-SA 4.0 cùng Data Access License Agreement.
