# Đánh giá cảnh báo theo đợt — phần B

**Hồi cứu trên validation cũ; không train, không đọc test, không đổi model được chọn.**

## 1. Quy tắc chốt trước khi tính

- Ngưỡng nghiên cứu 10 cm. Một đợt gồm ≥1 nhãn proc tương lai cao liên tiếp trên lưới quyết định 5 phút.
- Không gộp qua mẫu thấp hoặc khoảng thiếu. Báo rõ đợt thiếu quan sát đầu/cuối; không coi là trận mưa độc lập.
- Cảnh báo chính bật ngay khi dự báo ≥10 cm, tắt ở mẫu thấp/thiếu. Không cooldown, hysteresis hoặc gộp tùy ý.
- Đối chiếu đúng slot dự báo: cảnh báo chỉ khớp đợt nếu hoạt động ở cửa sổ có nhãn cao thuộc đợt. Một cảnh báo dài có thể khớp nhiều đợt và một đợt có thể có nhiều lần bật; báo riêng độ phân mảnh.
- Đợt cảnh báo không khớp nhãn cao nào là đợt báo giả theo archive. Cảnh báo vừa đúng vừa sai báo riêng số cửa sổ sai.
- Mẫu số là tổng cửa sổ hợp lệ ×5 phút quy ra ngày-trạm quan sát tương đương, không phải tổng ngày lịch.

## 2. Kết quả chính — bật từ một mẫu cao

| Phương pháp | Đợt nhãn cao | Phát hiện | Bỏ sót | Đợt báo giả | Báo giả/100 ngày-trạm quan sát | Đợt phát hiện bị bật/tắt nhiều lần |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 67 | 30 | 37 | 23 | 7.18 | 2 |
| linear_trend_5min | 67 | 40 | 27 | 153 | 47.78 | 9 |
| three_d3_weight20 | 67 | 39 | 28 | 145 | 45.29 | 2 |
| expanded_d3_weight20 | 67 | 35 | 32 | 64 | 19.99 | 4 |

Độ phủ: **92,215 cửa sổ**, **320.19 ngày-trạm quan sát tương đương**.
Trong 67 đợt, có **55** đợt có mẫu thấp liền trước và **53** đợt quan sát đủ cả hai đầu. Không tính một mẫu đầu chuỗi đang cao là một lần khởi phát đã biết.

## 3. Phát hiện sớm: chỉ ước lượng theo archive

Với đợt có mẫu thấp liền trước, thời điểm vượt ngưỡng nằm giữa timestamp nhãn thấp cuối và nhãn cao đầu. Lấy hai đầu khoảng đó trừ thời điểm quyết định khớp đầu tiên để ra khoảng báo trước. Không lấy lúc bắt đầu một cảnh báo vốn đã báo giả trước đó để kéo dài lead time.

“Sớm chắc theo khoảng” là cận dưới >0; vẫn chỉ theo đồng hồ proc archive, không phải xác nhận cảnh báo sớm ngoài thực tế. Trung vị chỉ ở các đợt phát hiện được có onset quan sát, không đại diện những đợt bỏ sót.

| Phương pháp | Onset quan sát được / phát hiện | Sớm chắc theo khoảng | Có thể sớm | Muộn chắc | Trung vị cận dưới / trên (phút, chỉ nhóm phát hiện) |
|---|---:|---:|---:|---:|---:|
| persistence | 55 / 18 | 4 | 4 | 14 | -5.90 / -0.66 |
| linear_trend_5min | 55 / 28 | 16 | 21 | 7 | 3.45 / 8.94 |
| three_d3_weight20 | 55 / 27 | 15 | 20 | 7 | 3.22 / 8.88 |
| expanded_d3_weight20 | 55 / 23 | 9 | 15 | 8 | -0.43 / 4.37 |

## 4. Độ nhạy cố định: chờ hai dự báo cao liên tiếp

Chỉ bật ở dự báo cao thứ hai, không ghi lùi thời điểm cảnh báo về mẫu thứ nhất. Đây là đối chiếu chính sách định trước, không tự chọn cấu hình sau khi xem điểm.

| Phương pháp | Phát hiện / tổng đợt | Bỏ sót | Đợt báo giả | Báo giả/100 ngày-trạm quan sát |
|---|---:|---:|---:|---:|
| persistence | 20 / 67 | 47 | 10 | 3.12 |
| linear_trend_5min | 24 / 67 | 43 | 17 | 5.31 |
| three_d3_weight20 | 25 / 67 | 42 | 33 | 10.31 |
| expanded_d3_weight20 | 24 / 67 | 43 | 17 | 5.31 |

## 5. Cách đọc và giới hạn

- Các đợt một mẫu không bị xóa. `event_metrics.csv` có thêm số đợt ≥2 mẫu và mức phát hiện của tập con đó.
- Báo giả ở đây là so với proc archive, không có kiểm định thực địa. Sự kiện cùng trạm/thời tiết có thể tương quan.
- Khoảng thiếu tách đợt và cảnh báo; tỷ lệ theo độ phủ không sửa được thiên lệch do mất mẫu. Đợt bị cắt có thể là mảnh của cùng một lần ngập; xem cờ censoring trong ledger.
- Không có arrival time/bản sao pipeline lọc nguồn nên lead time không xác nhận dự báo trực tiếp 10 phút hoặc chuyển sang ESP32.
- Validation đã dùng nhiều lần; không dùng kết quả phần B để tự đổi tiêu chí chọn model đã chốt bằng MAE.

## 6. Artifacts và kiểm tra

- `protocol_before_evaluation.json`: quy tắc trước nạp dữ liệu.
- `event_metrics.csv`: theo phương pháp, chính sách và trạm, kèm tổng pooled.
- `events_local.csv`, `alarms_local.csv`: ledger giữ cục bộ; có cờ đầu/cuối bị cắt, match và khoảng lead.
- `replayed_window_metrics.csv`: tái lập chỉ số cửa sổ cũ; kết quả chính một mẫu khớp TP/FP/FN cũ.
- `input_provenance.json`, `artifact_validation.json`: checksum và kiểm tra bảo toàn dữ liệu/model.

**Giữ model raw ba mã làm ứng viên theo tiêu chí cũ; không triển khai hoặc đổi cấu hình cảnh báo thật.** Trước bước vận hành cần chốt mức bỏ sót/báo giả chấp nhận được và đánh giá trên dữ liệu xác nhận độc lập. Demo lịch sử có thể trình bày AI cạnh baseline, tách khỏi cảnh báo theo số đo ESP32.

Nguồn: FloodNet (NYU/CUNY). Dữ liệu và artifacts giữ cục bộ, tuân giấy phép nguồn.
