# Kiểm kê FloodNet cục bộ

Thư mục này chứa công cụ và thống kê dẫn xuất. Không thay đổi CSV gốc, không train model, không tải lên dịch vụ bên ngoài. Các đầu ra vẫn cần được xem xét theo thỏa thuận FloodNet trước khi công khai hoặc bàn giao; không mặc định mọi artifact được phép phân phối.

## Chạy lại

```powershell
python "D:\CODE\IOT\IOT training\audit\audit_floodnet.py"
```

Chỉ dùng thư viện chuẩn Python. Đọc từng tháng, nhóm theo `deployment_id`, sắp xếp thời gian trong nhóm. Độ sâu sử dụng đơn vị mm. Thời gian UTC.

## Định nghĩa đợt ứng viên

- Dùng `depth_proc_mm`; chưa xác minh bộ lọc có tính nhân quả hoặc độ trễ bao nhiêu.
- Quét riêng các ngưỡng 10, 100, 200 mm. Đây là các phân tích nhạy cảm lồng nhau, KHÔNG cộng ba mức thành tổng sự kiện.
- Một đoạn vượt ngưỡng kết thúc khi có mẫu dưới ngưỡng, thiếu độ sâu xử lý, hoặc khoảng cách giữa hai mẫu vượt `max(180 giây, 3 × chu kỳ trung vị của trạm trong tháng)`.
- Đợt có duy trì (`sustained`) cần ít nhất 3 mẫu khác thời điểm và khoảng cách từ mẫu đầu đến cuối ít nhất 2 phút. Đây chỉ là quy tắc sàng lọc, không xác nhận ngập thực.
- Thời lượng chỉ tính từ mẫu đầu đến mẫu cuối, không suy ra nước liên tục giữa các mẫu. Trạm chu kỳ 5 phút bị sàng lọc thời lượng khác trạm 1 phút; phải chú ý khi so sánh.
- Bản ghi cùng trạm/cùng thời điểm không được tính thêm mẫu. Khi xung đột, giữ mẫu đầu sau sắp xếp ổn định và ghi số xung đột để kiểm tra riêng.
- `left_censored`: không có quan sát hợp lệ sát trước lúc bắt đầu. `end_reason` là `missing`, `gap`, hoặc `end_of_dataset` cho biết kết thúc chưa được xác nhận bằng mẫu dưới ngưỡng.
- Các đợt `pluvial` duy trì ở ngưỡng 100 mm được nhóm theo thời gian, gộp nếu cách cụm trước tối đa 6 giờ. Đây KHÔNG phải xác minh cùng cơn bão hoặc các sự kiện khí tượng độc lập. Chuỗi gộp có thể kéo dài; không dùng trực tiếp làm chia tập nếu chưa xem xét thời tiết.

## Đầu ra

- `summary.json`: tổng quan và định nghĩa phương pháp.
- `files.csv`: kích thước, số hàng, lỗi thời gian, số trạm từng file.
- `station_month_quality.csv`: chu kỳ, khoảng thiếu, mẫu thiếu/âm, chênh lệch raw/proc theo trạm-tháng.
- `stations.csv`: tổng hợp theo lần triển khai cảm biến.
- `candidate_episodes.csv`: phân tích các đoạn vượt ngưỡng, có cả đoạn ngắn chưa đạt điều kiện duy trì.
- `temporal_clusters.csv`: cụm thời gian của các đợt pluvial duy trì >=100 mm.

Giá trị cực đại, số vượt ngưỡng và độ dốc bất thường không tự động là ngập thật. Chênh lệch raw/proc không tự động là bộ lọc sai. Không dùng số đo tại một điểm để suy ra cả tuyến đường an toàn.

## Kiểm thử

```powershell
cd "D:\CODE\IOT\IOT training\audit"
python -m unittest test_audit_floodnet.py -v
```

Các kiểm thử chỉ sử dụng dữ liệu tổng hợp được đánh dấu rõ, không chứa dữ liệu FloodNet gốc.

## Nguồn và quyền sử dụng

FloodNet (New York University and The City University of New York).

Tuân theo Data Access License Agreement áp dụng với bản dữ liệu đã tải; chỉ phi thương mại theo thỏa thuận được cung cấp. Không phân phối CSV gốc, kể cả một phần, nếu chưa có chấp thuận bằng văn bản. Các phân tích/biểu diễn phái sinh được phép phải tuân CC BY-NC-SA 4.0 và điều khoản bổ sung. Không suy rằng trọng số model được phép công khai chỉ vì code train có thể chia sẻ.
