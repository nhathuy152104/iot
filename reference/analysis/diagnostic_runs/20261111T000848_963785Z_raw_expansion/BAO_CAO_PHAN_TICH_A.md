# Phân tích tác động mở rộng dữ liệu raw — phần A

**Chỉ phân tích hồi cứu; không train lại, không đọc test, không chọn lại model.**

## 1. Tỷ trọng train và trọng số

| Nhóm | Cửa sổ | Nhãn ≥10 cm | Tỷ lệ nhãn cao | Phần tổng trọng số ×20 của nhãn cao |
|---|---:|---:|---:|---:|
| core | 270,312 | 1,220 | 0.45% | 8.31% |
| added | 271,905 | 6,569 | 2.42% | 33.12% |
| expanded | 542,217 | 7,789 | 1.44% | 22.57% |

Trọng số ×20 giữ nguyên nhưng phân bố đóng góp vào hàm mất mát đổi đáng kể. Không suy ra quan hệ nhân quả hoặc hướng thay đổi dự báo chỉ từ tỷ trọng này. Các cấu hình uniform cũng kém hơn sau mở rộng, nên không thể quy toàn bộ suy giảm cho ×20.

## 2. Phân rã sai số trên cùng validation

Chênh MAE cân bằng (mở rộng − ba mã): **0.045239 cm**.

Đóng góp dưới đây dùng mẫu số toàn validation trong từng nhóm nhãn cao/thấp; cộng các hàng của mỗi chiều sẽ khớp chênh MAE cân bằng. Không lấy trung bình MAE cân bằng các trạm.

| Chiều | Nhóm | Cửa sổ | Nhãn cao | Đóng góp chênh MAE cân bằng (cm) |
|---|---|---:|---:|---:|
| station | daily_new_falcon | 51,444 | 8 | +0.003488 |
| station | ugliest_cyan_elephant | 40,771 | 1,029 | +0.041751 |
| phase | down_crossing | 70 | 0 | -0.000653 |
| phase | falling_gt_1cm | 955 | 64 | +0.007470 |
| phase | new_crossing | 83 | 83 | -0.006551 |
| phase | rising_gt_1cm | 633 | 82 | +0.000740 |
| phase | stable_high | 808 | 808 | +0.026283 |
| phase | stable_low | 89,666 | 0 | +0.017950 |
| target_band | [0.0, 1.0) | 81,977 | 0 | +0.020044 |
| target_band | [1.0, 10.0) | 9,201 | 0 | -0.003753 |
| target_band | [10.0, 20.0) | 884 | 884 | +0.019475 |
| target_band | [20.0, 50.0) | 153 | 153 | +0.009473 |

## 3. Giảm báo giả có đổi lấy bỏ sót không?

- Báo giả được sửa: **135**; báo giả mới: **22**.
- Bỏ sót được sửa: **1**; bỏ sót mới: **14**.
Đây là ghép cặp đúng cùng cửa sổ, không phải số đợt hay số thông báo.

## 4. Khác biệt dữ liệu bổ sung

Xem `feature_distributions.csv`, `train_composition.csv` và `train_raw_proc_comparison.csv`. So sánh raw/proc ở cùng bin train: raw là trung vị bin, proc là hàng cuối; chênh lệch phản ánh cả tổng hợp theo thời gian lẫn xử lý nguồn, không tự chứng minh lỗi hiệu chuẩn/nhãn.

| Mã | Tỷ lệ lịch sử có raw âm | Tỷ lệ lịch sử thưa | P99 chênh raw/proc tuyệt đối (cm) |
|---|---:|---:|---:|
| daily_new_falcon | 75.03% | 46.50% | 0.700 |
| widely_proud_lizard | 77.23% | 100.00% | 0.800 |
| ugliest_cyan_elephant | 49.81% | 100.00% | 1.100 |
| weekly_poetic_guinea | 57.26% | 75.00% | 1.000 |
| daily_gentle_beetle | 62.68% | 34.75% | 43.081 |
| overly_heroic_squid | 82.97% | 0.22% | 1.000 |
| simply_pet_joey | 9.95% | 44.12% | 72.000 |
| evenly_divine_dingo | 52.55% | 66.48% | 1.500 |
| weekly_fair_dove | 72.31% | 31.37% | 11.000 |
| tired_red_monkey | 74.64% | 82.09% | 30.015 |

## 5. Các ngày đóng góp tăng sai số nhiều nhất

Đây là xếp hạng hồi cứu để xem lỗi, không phải sự kiện ngập độc lập và không dùng để xóa mẫu.

- `ugliest_cyan_elephant_2023-06-02`: +0.015020 cm.
- `ugliest_cyan_elephant_2023-04-23`: +0.010607 cm.
- `ugliest_cyan_elephant_2023-03-04`: +0.006679 cm.
- `ugliest_cyan_elephant_2023-06-01`: +0.004260 cm.
- `ugliest_cyan_elephant_2023-01-26`: +0.003587 cm.

## 6. Biểu đồ và kiểm tra

- [ugliest_cyan_elephant_2023-06-02.png](ugliest_cyan_elephant_2023-06-02.png)
- [ugliest_cyan_elephant_2023-04-23.png](ugliest_cyan_elephant_2023-04-23.png)
- [ugliest_cyan_elephant_2023-03-04.png](ugliest_cyan_elephant_2023-03-04.png)
- [daily_new_falcon_2023-04-30.png](daily_new_falcon_2023-04-30.png)
- [ugliest_cyan_elephant_2023-06-12.png](ugliest_cyan_elephant_2023-06-12.png)
- [ugliest_cyan_elephant_2023-06-27.png](ugliest_cyan_elephant_2023-06-27.png)

- Tái lập 8 model + 2 baseline trên 92.215 cửa sổ: PASS.
- 56 file trong hai run cũ không đổi checksum.
- Không đọc test hoặc validation trạm mới; chỉ đọc cache các tháng trước 01/2023.
- Các ngưỡng pha/nhóm chỉ để mô tả hồi cứu; không là feature hoặc bộ lọc vận hành.

## 7. Kết luận và hướng tiếp

Giữ model ba mã theo tiêu chí cũ. Đã định vị các nhóm đóng góp tăng/giảm sai số; chưa tách được nguyên nhân do thành phần trạm, nhiễu, mốc đo hay động lực nước. Muốn kiểm chứng nhân quả cần thí nghiệm thay một yếu tố có kiểm soát, không chọn trạm bằng điểm validation. Bước tiếp theo là phần B: định nghĩa và đánh giá cảnh báo theo đợt. Chưa triển khai cảnh báo AI thật.

Nguồn: FloodNet (NYU/CUNY). Dữ liệu và artifacts giữ cục bộ; tuân giấy phép nguồn.
