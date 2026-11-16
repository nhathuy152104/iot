# Kết quả đánh giá cảnh báo theo đợt — phần B

**Ngày:** 11/11/2026. **Không train lại, không đọc test, không đổi lựa chọn model/cảnh báo thật.**

Run: [20261111T004533_510057Z_warning_events](diagnostic_runs/20261111T004533_510057Z_warning_events/BAO_CAO_CANH_BAO_THEO_DOT.md).

## 1. Định nghĩa trước khi tính

- Giữ ngưỡng nghiên cứu **10 cm**. Đợt nhãn cao = một hoặc nhiều cửa sổ có proc tương lai ≥10 cm liên tiếp trên lưới quyết định 5 phút.
- Không nối qua nhãn thấp hoặc khoảng thiếu; không fill, không xóa đợt một mẫu. Đây là đoạn quan sát vượt ngưỡng, **không phải các trận mưa/ngập độc lập đã xác minh**.
- Cảnh báo chính bật khi dự báo ≥10 cm và tắt ở mẫu thấp/thiếu. Cảnh báo chỉ khớp đợt khi hoạt động đúng slot có nhãn cao; không được tính đúng chỉ vì nằm gần đợt ngập.
- Một đợt cảnh báo có thể khớp nhiều đoạn nhãn cao; một đoạn có thể có nhiều lần bật cảnh báo. Có bảng kiểm tra phân mảnh, không giả định ghép một-một.
- Đợt báo giả = cả đợt không khớp nhãn cao nào. Đợt vừa đúng vừa sai không bị tính toàn bộ là báo giả; các cửa sổ sai vẫn được đếm riêng.
- Mẫu số tỷ lệ là tổng cửa sổ hợp lệ ×5 phút quy ra ngày-trạm quan sát tương đương, không phải số ngày lịch.
- Đối chiếu độ nhạy cố định: chờ **hai** dự báo cao liên tiếp, chỉ bật tại mẫu thứ hai. Không tự chọn hoặc tối ưu chính sách sau khi xem điểm.

## 2. Kết quả chính

Có **67 đoạn nhãn cao**, gồm **33 đoạn có ≥2 mẫu cao** và **34 đoạn chỉ một mẫu**. Trong 67 đoạn, 55 có mẫu thấp liền trước, 53 quan sát đủ cả hai đầu; còn lại có cờ thiếu quan sát đầu/cuối. Có 64 đoạn tại `ugliest_cyan_elephant` và ba đoạn tại `daily_new_falcon`.

Độ phủ: **92.215 cửa sổ**, tương đương **320,19 ngày-trạm quan sát** trên hai trạm.

| Phương pháp | Phát hiện / 67 đoạn | Bỏ sót | Đợt báo giả | Báo giả/100 ngày-trạm quan sát |
|---|---:|---:|---:|---:|
| Giữ nguyên độ sâu — persistence | 30 | 37 | 23 | 7,18 |
| Ngoại suy xu hướng 5 phút | 40 | 27 | 153 | 47,78 |
| Raw delta ba mã | 39 | 28 | 145 | 45,29 |
| Raw delta mười mã | 35 | 32 | 64 | 19,99 |

**AI ba mã phát hiện nhiều hơn baseline persistence chín đoạn, nhưng tạo thêm 122 đợt báo giả.** Ngoại suy xu hướng cũng đạt mức phát hiện tương tự AI, đổi lại nhiều báo giả hơn. Chưa chứng minh AI có ưu thế toàn diện cho cảnh báo.

Riêng nhóm đoạn có ≥2 mẫu cao, số phát hiện lần lượt **23/33, 28/33, 27/33, 26/33**. Đây là mô tả tập con, không thay điểm chính bằng cách bỏ các đoạn ngắn.

Với model ba mã: 185 đợt cảnh báo quan sát, gồm 145 đợt không khớp nhãn cao nào và 40 đợt có khớp; trong nhóm có khớp, 24 đợt vẫn chứa cửa sổ báo giả. Tổng cửa sổ báo giả vẫn đúng **246**, bỏ sót **57**, khớp báo cáo trước. Hai đoạn được phát hiện có nhiều lần bật cảnh báo.

## 3. Thời gian báo trước chỉ là ước lượng hồi cứu

Đối với đoạn có mẫu thấp liền trước, lấy khoảng timestamp giữa nhãn thấp cuối và nhãn cao đầu, rồi trừ thời điểm quyết định khớp đầu tiên. Không lấy thời điểm bắt đầu một cảnh báo vốn đã sai ở các slot trước đó để tăng thời gian báo trước.

Trong **55 đoạn có đầu đoạn quan sát được**:

| Phương pháp | Phát hiện | Cận dưới khoảng báo trước >0 | Cận trên >0 |
|---|---:|---:|---:|
| Persistence | 18 | 4 | 4 |
| Xu hướng | 28 | 16 | 21 |
| Raw delta ba mã | 27 | 15 | 20 |
| Raw delta mười mã | 23 | 9 | 15 |

Hai cột cuối lồng nhau, không cộng lại. Với model ba mã, trung vị cận dưới/trên ở **27 đoạn đã phát hiện** là **3,22 / 8,88 phút**; không áp dụng cho 28 đoạn onset quan sát nhưng chưa phát hiện.

Đây chỉ là khoảng chuyển ngưỡng giữa các mẫu proc archive. Không có thời điểm nhận bản tin/khả dụng của proc, không chứng minh cảnh báo thật sớm đúng từng ấy phút, càng không bảo đảm trước 10 phút hoặc chuyển tốt sang ESP32.

## 4. Chờ hai mẫu cao: giảm báo giả nhưng bỏ sót thêm

| Phương pháp | Phát hiện / 67 đoạn | Bỏ sót | Đợt báo giả |
|---|---:|---:|---:|
| Persistence | 20 | 47 | 10 |
| Xu hướng | 24 | 43 | 17 |
| Raw delta ba mã | 25 | 42 | 33 |
| Raw delta mười mã | 24 | 43 | 17 |

Với model ba mã: báo giả **145→33**, nhưng phát hiện **39→25**. Cách xác nhận này cũng làm trễ cảnh báo; không tự áp dụng vào ứng dụng chỉ vì số báo giả giảm.

## 5. Kiểm tra và artifacts

- **70 kiểm thử đạt:** 54 trong `analysis/` (bảy kiểm thử sự kiện mới) và 16 pipeline/đối soát 5 phút.
- Tái lập dự đoán và chỉ số cửa sổ của hai model + hai baseline trên toàn validation: PASS.
- Validator độc lập chỉ đọc: **24 hàng chỉ số** khớp ledger sự kiện/cảnh báo, số tổng theo trạm khớp pooled, FP/FN chế độ một mẫu khớp kết quả cũ.
- Checksum đầu vào/đầu ra/code khớp; **56 file** trong hai run raw cũ không đổi. Không sửa CSV/cache, model/provenance cũ, backend/frontend.
- Không nạp bảng train, test hoặc cache; các file run cũ, gồm train artifacts, chỉ được hash để kiểm tra không thay đổi.

Artifacts trong thư mục run:

- `protocol_before_evaluation.json`: quy tắc đã chốt trước nạp validation.
- `event_metrics.csv`: chỉ số theo trạm và tổng, một/hai mẫu xác nhận.
- `events_local.csv`, `alarms_local.csv`: ledger cục bộ có cờ thiếu đầu/cuối và khoảng lead.
- `replayed_window_metrics.csv`: đối chiếu chỉ số cửa sổ trước đây.
- `input_provenance.json`, `artifact_validation.json`: checksum và kiểm tra.

```powershell
# Tạo run mới; không train và không ghi đè run cũ.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\evaluate_warning_events.py"

# Chỉ kiểm tra run đã có; không fit hoặc ghi file.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\validate_warning_events.py" "D:\CODE\IOT\IOT training\analysis\diagnostic_runs\20261111T004533_510057Z_warning_events"
```

## 6. Kết luận và bước tiếp theo

Giữ model ba mã là **ứng viên nghiên cứu được chọn theo MAE**, không đồng nghĩa đã được chọn làm cảnh báo vận hành. Kết quả B cho thấy lợi ích phát hiện phải đánh đổi bằng số đợt báo giả lớn hơn baseline.

Trước thí nghiệm/tích hợp tiếp, cần chốt mục tiêu: ưu tiên sai số độ sâu hay cảnh báo sớm, mức báo giả/bỏ sót chấp nhận được, và tập xác nhận độc lập. Không tối ưu tiếp trên validation đã dùng nhiều lần rồi coi là bằng chứng xác nhận. Có thể chuẩn bị demo phát lại có AI cạnh baseline, tách khỏi ESP32 trực tiếp; chưa làm bước tích hợp trong phần B.

Nguồn: FloodNet (NYU/CUNY). Giữ dữ liệu/artifacts cục bộ; tuân giấy phép nguồn.
