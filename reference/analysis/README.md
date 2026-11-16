# Đối soát và chuẩn bị dữ liệu cục bộ — chưa train model

Nguồn: **FloodNet (New York University and The City University of New York).**

Giữ mọi đầu ra cục bộ. `local_cache` và `provisional_windows` chứa số liệu trích xuất/biến đổi từ nguồn; KHÔNG đưa lên GitHub hoặc gửi bên thứ ba khi chưa làm rõ quyền. Các phân tích/biểu diễn phái sinh được phép phải tuân CC BY-NC-SA 4.0 cùng điều kiện bổ sung của Data Access License Agreement. Không sửa CSV gốc.

## Chạy

```powershell
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\prepare_data.py"
```

Cần pandas, numpy, matplotlib, pyarrow. Bộ dữ liệu đầu ra là **tạm thời**, không chứng minh model dùng được thời gian thực. Chưa xác minh bộ lọc proc có nhìn tương lai. Cache phục vụ chạy lại và chỉ chứa ba trạm đã chọn; nếu thay nguồn hoặc chọn trạm khác, cần vô hiệu hóa cache tương ứng một cách có chủ đích.

## Ý nghĩa đầu ra

- `plots/`: đối chiếu raw/filt/proc trong cửa sổ bốn giờ quanh hai đỉnh proc, một ngày chênh lệch raw/proc lớn và một ngày proc bằng 0 ở mỗi trạm. Chọn để kiểm tra trường hợp khó, KHÔNG đại diện ngẫu nhiên cho toàn dữ liệu.
- `plot_cases.csv`: thời điểm, cực đại và độ thiếu trong từng biểu đồ.
- `local_cache/`: quan trắc ba trạm để phân tích cục bộ.
- `provisional_windows/`: Parquet theo trạm và split.
- `window_counts.csv`, `preparation_summary.json`: số cửa sổ, tỷ lệ nhãn cao, cờ đối soát, baseline persistence thăm dò.

Các cột `review_*` chỉ phục vụ đối soát, không phải đặc trưng model và không tự động xác nhận lỗi/ngập. Không dùng nhãn, thời điểm tương lai hoặc cờ được tính từ tương lai làm đặc trưng. Giữ metadata `deployment_id`, timestamp riêng với đặc trưng.

## Cửa sổ và chia tập

Lưới 1 phút lấy mẫu cuối có giá trị trong `(t-1 phút,t]` và gắn nhãn `t`. Không nội suy, không forward/backward fill. Mỗi trường được tổng hợp độc lập. Cần đủ 11 bin proc hợp lệ không âm từ `t-10 phút` đến `t`, và bin nhãn ở `t+10 phút` hợp lệ không âm. Không bắt buộc dữ liệu ở toàn bộ khoảng tương lai, vì bài toán là giá trị tại một thời điểm, không phải cực đại trong khoảng.

Train: toàn bộ lịch sử và nhãn trước 01/07/2022 UTC. Validation: toàn bộ support từ 01/07/2022 đến trước 01/01/2023 UTC. Test: toàn bộ support từ 01/01/2023 UTC. Loại cửa sổ chồng ranh giới, bao gồm cả khoảng bên trong bin sớm nhất `(t-11 phút,t-10 phút]`, không chỉ nhãn phải của bin. Đây là split cố định thăm dò, chưa chứng minh các trận mưa sát ranh giới đã độc lập. Không chỉnh model theo kết quả test. Trạm lịch sử ngắn có thể không có validation/test — không tự chia ngẫu nhiên để lấp chỗ trống.

Chính sách chặt này loại nhiều dữ liệu chu kỳ 5 phút và mất mẫu; đó không chứng minh các trạm đó vô dụng. Muốn dùng cần pipeline 5 phút riêng hoặc chính sách thiếu dữ liệu được thiết kế/kiểm định rõ.
