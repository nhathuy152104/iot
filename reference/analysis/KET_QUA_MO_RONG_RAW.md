# Kết quả mở rộng dữ liệu train nhánh raw

**Ngày:** 10/11/2026. **Run:** `training_runs/20261110T050356_109133Z_raw_expanded`.

**Kết luận: mở rộng chưa cải thiện MAE cân bằng trên validation cố định. Giữ model raw ba điểm làm ứng viên được chọn; không tự thay model hoặc triển khai cảnh báo.**

## 1. Dữ liệu và thiết kế

- Cả bảy ứng viên đều đạt các điều kiện độ phủ đã chốt bằng TRAIN: ≥1.000 cửa sổ, ≥30 ngày, ≥20 nhãn cao, ≥1.000 nhãn thấp, ≥3 ngày và ≥3 nhóm nhãn cao cách nhau hơn sáu giờ.
- Train core giữ nguyên **270.312** cửa sổ; bổ sung **271.905**; tổng **542.217**. Nhãn train ≥10cm tăng **1.220→7.789**.
- Số mã triển khai tăng từ 3 lên 10. Đây là `deployment_id`, không chắc là 10 vị trí độc lập; `widely_proud_lizard` và `overly_heroic_squid` đều được đặt tên 9th St/Smith St ở các giai đoạn khác nhau.
- Validation giữ nguyên đúng **92.215 cửa sổ**, cùng input raw và nhãn, gồm **1.037 nhãn ≥10cm**, tại hai trạm core cũ.
- Không đọc validation của điểm mới, không nạp test. Chỉ đọc cache ứng viên thuộc các tháng trước 01/2023 để kiểm kê/chuẩn bị train.
- Giữ nguyên xử lý raw 5 phút, nhãn proc tương lai, 12 feature, mốc chia tập/embargo, bốn cấu hình, seed và trọng số train. Không refit model ba điểm cũ.

### Dữ liệu bổ sung

| Mã triển khai | Cửa sổ train | Nhãn ≥10cm |
|---|---:|---:|
| daily_gentle_beetle | 46.075 | 136 |
| evenly_divine_dingo | 19.354 | 484 |
| overly_heroic_squid | 46.080 | 322 |
| simply_pet_joey | 20.549 | 2.430 |
| tired_red_monkey | 20.991 | 1.608 |
| weekly_fair_dove | 52.606 | 1.322 |
| weekly_poetic_guinea | 66.250 | 267 |

**Đạt độ phủ không phải nhãn sạch.** Các điểm từng có nghi vấn mức cao kéo dài, raw/proc khác mốc, raw âm hoặc dao động mạnh. Lần này giữ toàn bộ cửa sổ raw hợp lệ để thử nghiệm tác động của bổ sung dữ liệu; không dùng mask offline hay loại ca khó để cải thiện điểm. Nhóm sáu giờ chỉ là quy ước độ phủ, không là trận mưa độc lập đã xác minh.

## 2. Kết quả trên đúng cùng validation

| Chỉ số | Raw persistence | Raw delta 3 mã | Raw delta 10 mã |
|---|---:|---:|---:|
| MAE toàn tập cm | 0,1212 | **0,1029** | 0,1357 |
| MAE khi nhãn ≥10cm | 0,9500 | **0,8960** | 0,9539 |
| MAE cân bằng cm | 0,5309 | **0,4949** | 0,5402 |
| Cửa sổ báo giả | **70** | 246 | 133 |
| Cửa sổ bỏ sót | 83 | **57** | 70 |

Ứng viên tốt nhất của cả hai nhóm đều là cấu hình `d3_weight20`. MAE cân bằng của nhóm mở rộng tăng **0,04524 cm, khoảng 9,1%** so với nhóm ba mã. Báo giả giảm **246→133**, nhưng bỏ sót tăng **57→70**. Không đánh đồng việc giảm báo giả với cải thiện toàn diện.

Cả bốn cặp cấu hình giống nhau đều có MAE cân bằng kém hơn sau khi thêm dữ liệu. Theo tiêu chí đã chốt, phương pháp được chọn vẫn là **`three_d3_weight20`**, tương ứng model cũ `raw_delta_d3_weight20.ubj`. Không thay tiêu chí sau khi xem kết quả để ưu tiên nhóm mở rộng.

Kết quả này cho thấy **cách bổ sung dữ liệu đã thử chưa hữu ích cho validation hiện tại**, không chứng minh mọi phương án thêm trạm đều vô ích, cũng không chứng minh các nhãn thêm vào đều sai. Chưa tách được đầy đủ tác động khác trạm, hiệu chuẩn, nhiễu và diễn biến nước.

## 3. Kiểm tra kỹ thuật

- **57 kiểm thử đạt:** 41 kiểm thử trong `analysis/` và 16 pipeline/đối soát 5 phút; có tám kiểm thử mở rộng mới.
- Dựng lại cả bảy bộ cửa sổ ứng viên từ **62 cache có checksum**, xác minh **27 fingerprint CSV nguồn**, kiểm kê, lý do loại và quyết định chọn trạm khớp: **PASS**.
- Train core và validation vẫn khớp nguyên bản; tất cả checksum đầu vào/model đã đóng băng không đổi.
- Nạp lại tám model delta (bốn cũ, bốn mới) và hai baseline; tái lập mọi dự đoán/chỉ số trên 92.215 hàng validation: **PASS**.
- Bảng so từng cấu hình, từng trạm và quyết định chọn model khớp. Không fit lại khi kiểm tra.
- Không sửa CSV/cache, cửa sổ/model cũ, backend/frontend hoặc cảnh báo/định tuyến.

Các kiểm tra không chứng nhận nhãn ngập thật, dự báo thời gian thực hoặc tổng quát hóa sang các trạm mới/ESP32. Validation cũ đã được dùng nhiều lần, chưa là kiểm định xác nhận độc lập.

## 4. Artifacts và bước tiếp theo

- [Báo cáo đầy đủ](training_runs/20261110T050356_109133Z_raw_expanded/BAO_CAO_MO_RONG_RAW.md).
- [Lệnh chạy/kiểm tra](raw_expansion/README.md).
- `station_audit_train.csv`, `candidate_exclusion_reasons.csv`: độ phủ và chất lượng train, không giấu ca khó.
- `matched_config_comparison.csv`, `validation_breakdown.csv`: đối chiếu cấu hình/trạm.
- `artifact_validation.json`: xác nhận tái lập kỹ thuật.

**Khuyến nghị:** giữ model ba mã làm đối chứng; chưa tiếp tục thêm điểm hoặc đổi tham số chỉ để tìm điểm validation đẹp. Bước nghiên cứu tiếp theo nên phân tích sự khác biệt ở các nhóm raw/nhãn đã thêm và đánh giá báo giả/bỏ sót theo đợt trước khi quyết định một thí nghiệm mới. Không loại từng trạm dựa vào test hoặc chỉ vì model sai.

Nguồn: FloodNet (New York University and The City University of New York). Artifacts giữ cục bộ; làm rõ quyền trước phân phối dữ liệu/trọng số. Phân tích/biểu diễn phái sinh được phép theo CC BY-NC-SA 4.0 cùng Data Access License Agreement.
