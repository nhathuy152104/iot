# Kết quả train thử nghiệm baseline và XGBoost

**Chỉ nghiên cứu hồi cứu. Chưa được duyệt để cảnh báo thực tế; chưa kiểm chứng proc causal hoặc nhãn ngập thật.**

Model được chọn bằng validation: **d3_weight20**. Không refit bằng validation; model đó được lưu và kiểm tra nạp lại trước dự đoán test.
Số cửa sổ: {'train': 270313, 'validation': 92215, 'test': 42218}. Dùng toàn bộ cửa sổ gốc của 3 trạm ứng viên, không dùng mask đối soát offline để loại ca khó.

## Validation

| Model | MAE toàn tập (cm) | MAE nhãn >=10cm | MAE nhãn <10cm | MAE cân bằng |
|---|---:|---:|---:|---:|
| persistence | 0.0693 | 0.9104 | 0.0597 | 0.4851 |
| linear_trend_5min | 0.1176 | 1.1588 | 0.1057 | 0.6323 |
| d3_uniform | 0.0846 | 1.0916 | 0.0732 | 0.5824 |
| d3_weight20 | 0.1352 | 0.8994 | 0.1265 | 0.5130 |
| d4_uniform | 0.0841 | 1.0853 | 0.0727 | 0.5790 |
| d4_weight20 | 0.1226 | 0.9282 | 0.1134 | 0.5208 |

Tiêu chí chọn chỉ so 4 cấu hình XGBoost: MAE cân bằng nhỏ nhất trên validation, hòa thì MAE toàn tập và tên ID. Baseline vẫn có thể tốt hơn model được chọn; không ép kết luận XGBoost thắng. Trọng số 20 chỉ dùng train cho hai cấu hình, không đổi tỷ lệ mẫu validation/test.

## Test thăm dò — không dùng chọn cấu hình

| Model | MAE toàn tập (cm) | MAE nhãn >=10cm | MAE cân bằng | Recall >=10cm | Precision >=10cm | FP | FN | Recall chuyển từ <10 sang >=10cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| persistence | 0.0913 | 5.5073 | 2.7905 | 0.6861 | 0.6861 | 43 | 43 | 0.0000 |
| linear_trend_5min | 0.1436 | 9.0664 | 4.5905 | 0.7080 | 0.4236 | 132 | 40 | 0.2093 |
| xgboost_selected | 0.1848 | 14.8377 | 7.4874 | 0.7591 | 0.4031 | 154 | 33 | 0.2326 |

Test có 137 cửa sổ nhãn >=10 cm và 43 cửa sổ hiện tại <10 cm nhưng nhãn tương lai >=10 cm. Đây không phải số trận ngập độc lập.
Recall/precision/FP/FN là theo cửa sổ tại một bin tương lai. Không phải lead time cảnh báo trước khởi phát và không phải xác suất model đã được hiệu chuẩn.

## Giới hạn bắt buộc

- Nhãn là số đo cuối trong (t+5,t+10 phút], không phải độ sâu tức thời đúng t+10. Baseline xu hướng ngoại suy tới nhãn bin +10 phút, có biến động lead time thực tế.
- Bộ lọc proc chưa được xác minh tính nhân quả. Tổng hợp dữ liệu causal không khắc phục được bộ lọc archive không causal.
- Trạm widely_proud_lizard có ca nghi box; raw/proc và nhãn chưa được xác minh hiện trường. Không tự coi nhãn xử lý là ground truth.
- Dữ liệu 2023 đã từng được xem thống kê/biểu đồ. Test này chỉ thăm dò, không là holdout xác nhận độc lập chưa từng xem.
- Phân chia thời gian và embargo giữ nguyên từ pipeline 5 phút; tính mùa vụ, khác trạm và số nhãn cao ít có thể chi phối kết quả.
- Chưa bổ sung 7 trạm cần xem xét, chưa dùng dữ liệu ESP32, chưa tích hợp API/frontend.
- Không chỉ báo cáo MAE ngày khô; xem cả MAE lúc cao, bỏ sót, báo giả, các lần chuyển ngưỡng và breakdown theo trạm/quý.

## Artifacts

- `xgboost_selected.ubj`: model cục bộ, không tự động phê duyệt deploy.
- `config_before_fit.json`, `selection_before_test.json`: cấu hình chốt trước fit và lựa chọn trước test.
- `metrics.csv`, `test_breakdown.csv`, `summary.json`: kết quả tổng hợp.
- `test_predictions_local.parquet`: đầu ra dẫn xuất dữ liệu; không công khai.
- `input_provenance.json`: đường dẫn/số hàng/checksum các file đã dùng.
- `feature_importance.csv`: mức quan trọng mô hình, không phải quan hệ nhân quả.
- `test_*.png`: biểu đồ trường hợp test chọn sau khi chốt model, chỉ dùng chẩn đoán.

Nguồn: FloodNet (New York University and The City University of New York). Giữ artifacts cục bộ. Không phân phối dữ liệu/trích xuất hoặc trọng số khi chưa làm rõ quyền. Phân tích/biểu diễn phái sinh được phép tuân CC BY-NC-SA 4.0 cùng Data Access License Agreement.
