# Nhánh raw — ba trạm, train/validation cố định

Nhánh độc lập, không sửa kết quả proc hoặc backend/frontend. Dùng raw làm đầu vào, proc **chỉ làm nhãn tương lai** và đối chứng hồi cứu. Không tự động phê duyệt model để cảnh báo hay định tuyến.

## Chạy

```powershell
# Mỗi lần tạo training_runs/<timestamp>_raw_delta riêng.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\run_relocated.py" "D:\CODE\IOT\IOT training\analysis\train_raw_delta.py"

# Dựng lại cửa sổ và kiểm tra model đã lưu; không refit, không đọc test.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\run_relocated.py" "D:\CODE\IOT\IOT training\analysis\validate_raw.py" "D:\CODE\IOT\IOT training\analysis\training_runs\20261110T043429_532355Z_raw_delta"
```

Đọc kết quả: [KET_QUA_NHANH_RAW.md](../KET_QUA_NHANH_RAW.md).

## Quy tắc không được thay ngầm

- Đúng ba trạm core của lượt đầu, không thêm bảy trạm hoặc xếp hạng lại bằng validation.
- Bước 5 phút. Trung vị raw trong bin vừa kết thúc, không dùng bin tương lai. Các bin không đủ thời gian kết thúc ở cuối nguồn không được sử dụng.
- Raw âm quy về 0 cho đầu vào, đồng thời giữ cờ/số lượng. Không tuyên bố raw âm là lỗi hoặc chắc chắn không ngập.
- Nếu hàng vật lý cuối của bin có raw null/không hữu hạn, bin thiếu; không quay lại lấy hàng tốt trước đó.
- Bin rỗng vẫn thiếu. Không nội suy, carry-forward/backward, cắt trên hoặc loại mọi nước dâng nhanh.
- Số mẫu tối thiểu mỗi bin là một để giữ trạm đo thưa. Vì vậy trung vị không xử lý được blip trong bin một mẫu hay vật cản kéo dài.
- Lịch sử: ba bin t−10/t−5/t, 12 feature như trước. Nhãn: proc cuối bin (t+5,t+10], không phải cực đại hoặc đúng tức thời t+10.
- Giữ mốc 01/01/2023, 01/07/2023 và embargo 24h; không dùng test/cache từ 07/2023 trở đi.
- Giữ bốn cấu hình XGBoost/seed/trọng số của nhánh delta trước. Trọng số theo nhãn tương lai, không theo delta.
- Chọn bằng MAE cân bằng trên toàn validation raw-eligible. Baseline được phép thắng. Đối chiếu giao raw/proc chỉ sau khi chốt.

## Giới hạn quan trọng

“Chỉ dùng quá khứ” ở đây được kiểm tra theo timestamp quan trắc trong archive. Không có thời điểm nhận raw để tái dựng độ trễ mạng hoặc xác minh toàn bộ hiệu chuẩn nguồn. Mốc đo thực tế, thời điểm nhận, thời điểm hoàn tất xử lý và thời điểm phát hành dự báo phải được ghi rõ khi tích hợp sau này.

Proc tương lai là nhãn tham chiếu hồi cứu chưa xác nhận hiện trường; nó không đi vào đặc trưng hoặc điều kiện lịch sử raw. Chưa chứng minh khả năng dự báo trực tiếp trước 10 phút hoặc chuyển từ NYC sang trạm ESP32.

Nguồn: FloodNet (New York University and The City University of New York). Không đưa cache/cửa sổ/dự đoán/trọng số lên kho chung hoặc gửi bên thứ ba khi chưa làm rõ quyền. Phân tích/biểu diễn phái sinh được phép theo CC BY-NC-SA 4.0 cùng Data Access License Agreement.
