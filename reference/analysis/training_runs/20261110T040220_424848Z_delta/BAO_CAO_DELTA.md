# Thí nghiệm XGBoost dự báo mức thay đổi

**Chỉ đánh giá validation. Không nạp/dự đoán test; chưa duyệt triển khai.**

Giữ nguyên dữ liệu, 3 trạm core, 12 feature, 4 cấu hình, seed42, trọng số và hàm mục tiêu squared error. Chỉ đổi đích học từ độ sâu thành `độ sâu tương lai − độ sâu hiện tại`; cộng lại độ sâu hiện tại trước khi chấm điểm.

Số cửa sổ: {'train': 270313, 'validation': 92215}. Không lọc bằng mask offline; không thêm 7 trạm.

| Model | MAE toàn tập | MAE ≥10cm | MAE <10cm | MAE cân bằng | Recall | Precision | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| persistence | 0.0693 | 0.9104 | 0.0597 | 0.4851 | 0.9238 | 0.9247 | 78 | 79 |
| linear_trend_5min | 0.1176 | 1.1588 | 0.1057 | 0.6323 | 0.9296 | 0.7651 | 296 | 73 |
| original_absolute_selected | 0.1352 | 0.8994 | 0.1265 | 0.5130 | 0.9431 | 0.7443 | 336 | 59 |
| delta_d3_uniform | 0.0783 | 0.9442 | 0.0685 | 0.5063 | 0.9161 | 0.9350 | 66 | 87 |
| delta_d3_weight20 | 0.1083 | 0.8450 | 0.0999 | 0.4725 | 0.9402 | 0.7506 | 324 | 62 |
| delta_d4_uniform | 0.0793 | 0.9425 | 0.0694 | 0.5060 | 0.9142 | 0.9349 | 66 | 89 |
| delta_d4_weight20 | 0.0982 | 0.8672 | 0.0895 | 0.4784 | 0.9335 | 0.8067 | 232 | 69 |

Ứng viên delta tốt nhất: **delta_d3_weight20**. Phương pháp tốt nhất trong toàn bộ so sánh: **delta_d3_weight20**.
Chênh lệch MAE cân bằng của delta tốt nhất so với persistence: **-0.012602 cm** (âm tốt hơn).

Đây là lựa chọn thăm dò trên validation đã dùng nhiều lần, không phải chứng minh tổng quát hóa. Không coi giảm một chỉ số là đã đạt yêu cầu cảnh báo; phải xem báo giả, bỏ sót và đánh giá theo sự kiện độc lập.

## Giới hạn

- Nhãn là số đo cuối bin (t+5,t+10], không phải chính xác t+10 hoặc cực đại 10 phút.
- Proc chưa được xác minh causal; nhãn chưa đối chiếu hiện trường. Không dùng kết quả để quyết định đường an toàn.
- Test cũ chỉ dùng trong báo cáo chẩn đoán riêng, không dùng chọn/tinh chỉnh thí nghiệm này. Cần bộ xác nhận độc lập sau này.
- Chưa chạy ESP32, API hoặc frontend; không thay cảnh báo theo ngưỡng hiện tại.

## Artifacts

- `config_before_fit.json`: cấu hình và checksum trước fit.
- `selection_validation_only.json`, `metrics.csv`, `summary.json`: quyết định và chỉ số.
- `delta_*.ubj`: cả bốn model nghiên cứu, không tự động duyệt deployment.
- `validation_predictions_local.parquet`, `input_provenance.json`: giữ cục bộ.

Nguồn: FloodNet (New York University and The City University of New York). Không phân phối dữ liệu, dự đoán hoặc trọng số khi chưa làm rõ quyền. Phân tích/biểu diễn phái sinh được phép: CC BY-NC-SA 4.0 cùng Data Access License Agreement.
