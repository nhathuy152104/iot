# Thí nghiệm mở rộng train cho nhánh raw

Giữ nguyên nhánh raw ba mã triển khai làm đối chứng; chỉ bổ sung dữ liệu train từ bảy ứng viên đã định trước. Không sửa nhánh cũ, không thêm feature hoặc đổi nhãn/preprocessing/trọng số, không đọc test.

Kết quả đã hoàn thành: [KET_QUA_MO_RONG_RAW.md](../KET_QUA_MO_RONG_RAW.md).

## Chạy

```powershell
# Tạo run mới và chỉ kiểm kê/chọn điểm bằng train. Chưa fit/chấm validation.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\run_relocated.py" "D:\CODE\IOT\IOT training\analysis\train_raw_expanded.py" --prepare-only

# Sau khi xem kiểm kê, fit trên run vừa chuẩn bị (thay đường dẫn bằng run mới).
# Không resume lượt bên dưới: nó đã fit và script sẽ từ chối ghi đè.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\run_relocated.py" "D:\CODE\IOT\IOT training\analysis\train_raw_expanded.py" --resume "<duong-dan-run-moi-vua-chuan-bi>"

# Kiểm tra lượt đã có, không fit hoặc chỉnh artifacts cũ.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\run_relocated.py" "D:\CODE\IOT\IOT training\analysis\validate_raw_expanded.py" "D:\CODE\IOT\IOT training\analysis\training_runs\20261110T050356_109133Z_raw_expanded"
```

Chạy `train_raw_expanded.py` không có tham số sẽ tạo run mới, chuẩn bị và fit liên tiếp. Không tự chạy lại khi chỉ cần đọc kết quả.

## Quy tắc đã chốt trước fit

- Ứng viên: `daily_gentle_beetle`, `evenly_divine_dingo`, `overly_heroic_squid`, `simply_pet_joey`, `tired_red_monkey`, `weekly_fair_dove`, `weekly_poetic_guinea`.
- Chỉ đọc cache ứng viên thuộc tháng trước 01/2023, cắt quan trắc theo embargo train. Không đọc validation các điểm mới hoặc test.
- Độ phủ tối thiểu: 1.000 cửa sổ, 30 ngày, 20 nhãn ≥10cm, 1.000 nhãn thấp, ba ngày và ba nhóm nhãn cao cách nhau hơn sáu giờ. Không điều chỉnh theo điểm model/validation.
- Đạt độ phủ KHÔNG xác nhận nhãn sạch. Đây là thử nghiệm dùng tất cả cửa sổ raw hợp lệ của điểm đạt, không dùng mask offline hoặc chỉ lấy đoạn đẹp.
- Giữ nguyên toàn bộ train core cũ và đúng 92.215 hàng validation/nhãn đã đóng băng. Source/train/preprocessing/feature/model configs có checksum.
- Giữ bốn cấu hình XGBoost delta. So sánh cả từng cặp cấu hình giống nhau và ứng viên tốt nhất mỗi nhóm. Cho phép baseline hoặc model ba điểm thắng.
- Không dùng kết quả để tự thay cảnh báo thật hoặc thay model cũ trên ứng dụng.

## Giới hạn

`deployment_id` là mã lần triển khai, không nhất thiết là một vị trí mới. Ví dụ `widely_proud_lizard` và `overly_heroic_squid` đều mang tên 9th St/Smith St ở hai giai đoạn khác nhau; không gọi mười mã là mười vị trí độc lập đã xác minh.

Raw đầu vào chỉ nhân quả theo thời gian archive, chưa biết toàn bộ độ trễ nhận/hiệu chuẩn. Nhãn proc tương lai vẫn là tham chiếu hồi cứu. Validation chỉ ở hai trạm core cũ, không xác nhận khả năng dự báo tại bảy trạm bổ sung. Các cửa sổ và nhóm sáu giờ không phải những trận ngập độc lập.

Nguồn: FloodNet (New York University and The City University of New York). Giữ dữ liệu, dự đoán, cửa sổ và trọng số cục bộ; làm rõ quyền trước phân phối. Phân tích/biểu diễn phái sinh được phép theo CC BY-NC-SA 4.0 cùng Data Access License Agreement.
