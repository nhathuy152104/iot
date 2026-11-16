# Thí nghiệm độ sâu: đổi hàm học sang MAE

**Một thí nghiệm cố định, hồi cứu trên validation đã dùng nhiều lần; không phê duyệt model mới.**

| Model | MAE cân bằng cm | MAE toàn tập cm | MAE nhãn ≥10cm |
|---|---:|---:|---:|
| persistence | 0.530877 | 0.121228 | 0.949952 |
| linear_trend_5min | 0.573433 | 0.161926 | 0.994407 |
| three_d3_weight20 | 0.494940 | 0.102866 | 0.896036 |
| trial_absolute_error | 0.494940 | 0.074745 | 0.924804 |

Chỉ đổi objective reg:squarederror → reg:absoluteerror; giữ d3/250 cây/weight20, train raw 270.312 và validation 92.215, feature/nhãn/mốc chia nguyên bản. Không thêm dữ liệu hoặc lọc ca khó.

Xem `depth_error_profiles.csv` để kiểm tra quanh ngưỡng 8–12 cm và P90/P99/sai lớn nhất. Nhóm theo nhãn và nhóm theo dự báo là hai tập khác nhau; không so các MAE nhóm theo dự báo như trên cùng cửa sổ. P90 lỗi hồi cứu không phải khoảng tin cậy cá nhân đã hiệu chuẩn.

Không thay model ba mã hoặc artifact demo theo kết quả thăm dò. Cần tập xác nhận độc lập cho kết luận cải thiện.

Nguồn: FloodNet NYU/CUNY. Artifacts giữ cục bộ; tuân giấy phép nguồn.
