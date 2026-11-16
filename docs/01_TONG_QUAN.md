# 01 — Tổng quan và trạng thái

## Bài toán

Hệ thống chính: ESP32 → MQTT → backend → bản đồ/cảnh báo. Do lịch sử ESP32 còn ít, pipeline AI được nghiên cứu bằng chuỗi cảm biến FloodNet NYC, không phải bộ ảnh FloodNet.

Hướng sản phẩm: hiển thị **độ sâu dự báo bằng cm**, xu hướng và mức gần ngưỡng. Cảnh báo nhị phân là lớp riêng, không dùng thay việc đánh giá sai số độ sâu. Trang `/forecast` hiện chỉ phát lại dữ liệu lịch sử NYC, tách khỏi ESP32/định tuyến thật.

## Model đang được giữ

`model/raw_delta_d3_weight20.ubj`, sao từ run `20261110T043429_532355Z_raw_delta`.

- XGBoost CPU, depth3, 250 cây, seed42, nhãn train ≥10 cm trọng số ×20.
- Ba deployment: `daily_new_falcon`, `ugliest_cyan_elephant`, `widely_proud_lizard`. Mã deployment không nhất thiết là vị trí độc lập.
- Train 270.312 cửa sổ; validation 92.215 cửa sổ tại hai deployment còn dữ liệu.
- Đích `.predict()` là **delta = proc tương lai − raw hiện tại**, không phải độ sâu trực tiếp.

```text
depth_future_cm = max(0, current_raw_median_cm + predicted_delta_cm)
```

## Hợp đồng dữ liệu

- UTC, bin 5 phút `(t−5,t]`, dùng trung vị raw hữu hạn sau khi raw âm quy về 0.
- Hàng vật lý cuối bin có raw null/không hữu hạn → bin thiếu, không dùng raw cũ thay thế.
- Lịch sử: t−10, t−5, t; support quan trắc thực tế `(t−15,t]`.
- Không fill/nội suy, không xóa mức cao/nhảy mạnh chỉ vì nghi nhiễu.
- 12 feature phải đúng tên/thứ tự theo `model/config_before_fit.json`.
- Nhãn khi train/đánh giá: proc cuối bin tương lai `(t+5,t+10]`, không phải cực đại 10 phút hoặc số đo chính xác tức thời t+10.
- Khi **inference**, chỉ cần lịch sử raw; KHÔNG cần hoặc đọc nhãn proc tương lai. Eligibility có nhãn của validation không được dùng làm điều kiện vận hành online.

## Kết quả đã biết (validation đã được dùng nhiều lần)

| Phương pháp | MAE cân bằng cm | MAE toàn tập cm | MAE nhãn ≥10 cm |
|---|---:|---:|---:|
| Persistence raw | 0,5309 | 0,1212 | 0,9500 |
| Raw ba mã đang chọn | 0,4949 | 0,1029 | 0,8960 |
| Raw mười mã | 0,5402 | 0,1357 | 0,9539 |
| Trial absolute-error | 0,49494045 | 0,0747 | 0,9248 |

MAE cân bằng = (MAE nhãn ≥10 cm + MAE nhãn <10 cm)/2. Model ba mã cũ đạt 0,49494021; trial gần như hòa nhưng hơi kém và sai số nhóm nước cao tăng. Không đổi sang trial chỉ vì MAE toàn tập đẹp hơn.

Cảnh báo theo đợt tại ngưỡng 10 cm: persistence phát hiện 30/67 đoạn và có 23 đợt báo giả; ba mã 39/67 và 145 báo giả; mười mã 35/67 và 64 báo giả. Các đoạn này không phải trận ngập độc lập đã xác minh. Tốt hơn về MAE không mặc nhiên tốt hơn toàn diện về cảnh báo.

Quanh nhãn 8–<12 cm, model cũ MAE 1,1803 cm, kém persistence 0,9899 cm. Có ca sai rất lớn dù MAE tổng nhỏ. Không diễn giải dự báo 9 cm là chắc chắn dưới 10 cm hoặc đường an toàn.

## Giới hạn phải truyền đạt

- Nhãn proc chưa được xác minh độc lập ngoài hiện trường. Bộ lọc nguồn có cơ chế chờ mẫu sau; archive không đủ thông tin arrival/availability để xác nhận lead time thật.
- Raw feature chỉ nhân quả theo đồng hồ archive; không tái dựng đầy đủ mạng/hiệu chuẩn.
- NYC chưa chứng minh chuyển sang địa bàn ESP32 tốt. Một điểm đo không đại diện cả đường.
- Test 2023 đã từng được khảo sát; không phải holdout xác nhận hoàn toàn chưa xem. Các thí nghiệm mới không tiếp tục dùng test để chọn model.
- Mọi kết quả ở đây là nghiên cứu/demo, chưa được duyệt cảnh báo hay định tuyến thật.
