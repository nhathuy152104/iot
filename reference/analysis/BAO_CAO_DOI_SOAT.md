# Đối soát raw/filt/proc và chuẩn bị cửa sổ dự báo

## Kết luận

Đã tạo 12 biểu đồ ở ba trạm và 9 file Parquet theo trạm/train-validation-test. Chưa train XGBoost. Cửa sổ chỉ là **dữ liệu thăm dò**, chưa chấp nhận làm bộ đánh giá dự báo ngập đáng tin cậy.

Không nên đi thẳng từ “17 triệu bản ghi” đến kết luận có nhiều ví dụ ngập hữu ích: chu kỳ đo, thiếu mẫu, bộ lọc và chia tập làm thay đổi đáng kể số cửa sổ sử dụng được.

## Đối soát biểu đồ

Đã trực tiếp xem 4/12 biểu đồ: daily_new_falcon ca đỉnh thứ nhất và ca chênh lệch raw/proc, tired_red_monkey ca đỉnh thứ nhất, simply_pet_joey ca đỉnh thứ nhất. Các biểu đồ còn lại được tạo và có thống kê tự động, chưa được xác minh thủ công.

- **daily_new_falcon, 14/02/2021:** raw/filt/proc trùng nhau ở các mẫu có mặt, cực đại 48,6 cm; chỉ 12/241 bin phút có proc trong đoạn bốn giờ. Không đủ mật độ cho cửa sổ liên tục một phút. “Bin phút không có mẫu” ở đây không đồng nghĩa cảm biến lỗi: trạm cũ có thể gửi thưa hơn. Không xác nhận đây là ngập thực.
- **daily_new_falcon, 24/09/2021:** raw có đỉnh 239,3 cm, proc vẫn 0 trong cửa sổ. Hình thái gợi ý một bước nhảy đã bị lọc, nhưng không có đối soát hiện trường để xác minh nguyên nhân.
- **tired_red_monkey, 22/10/2022:** raw có các mức gần 80 cm; proc có lúc giữ, có lúc loại. Có khoảng trống và bước chuyển đột ngột. Không nên coi số đoạn vượt ngưỡng lớn là nhiều trận ngập đã được xác nhận.
- **simply_pet_joey, 12/01/2022:** các mức khoảng 80–156 cm biến động mạnh; đầu đoạn proc bằng 0 trong khi raw/filt cao, rồi proc giữ nhiều mức cao. Cần điều tra vật cản, mốc nền, bộ lọc và bối cảnh; không tự kết luận toàn bộ là nhiễu hay toàn bộ là ngập.

Các trường hợp được chọn theo đỉnh/chênh lệch để kiểm tra khó khăn, không phải mẫu ngẫu nhiên đại diện. Chuỗi qua nhiều tháng/năm có thể thay mốc hiệu chuẩn. Chưa kiểm chứng tính nhân quả/độ trễ của bộ lọc proc.

## Cửa sổ đã tạo

- Tổng hợp thành bin một phút, lấy giá trị cuối có mặt trong `(t-1 phút,t]`, không dùng mẫu sau t.
- Không nội suy hoặc điền giá trị cũ vào khoảng thiếu.
- Cần 11 bin proc không âm, có đủ từ t-10 phút đến t.
- Nhãn là bin proc ở t+10 phút; không phải cực đại trong 10 phút và không phải khẳng định số đo tức thời chính xác ở giây 0.
- Các cột lag, tốc độ dâng/rút, trung bình/min/max/std chỉ dùng lịch sử.
- Cờ `review_*` là thông tin đối soát, không phải đặc trưng để train tự động và không tự loại mẫu.

### Chia tập cố định theo thời gian

- Train: toàn bộ support trước 01/07/2022 UTC.
- Validation: toàn bộ support từ 01/07/2022 đến trước 01/01/2023 UTC.
- Test: toàn bộ support từ 01/01/2023 UTC.
- Purge cả khoảng bên trong bin lịch sử sớm nhất, không chỉ timestamp nhãn bin. Chưa xác minh các sự kiện khí tượng sát ranh giới đã độc lập.

| Trạm | Train (nhãn >=10 cm) | Validation (nhãn >=10 cm) | Test (nhãn >=10 cm) |
|---|---:|---:|---:|
| daily_new_falcon | 111.383 (80) | 104.968 (16) | 165.330 (15) |
| tired_red_monkey | 0 (0) | 148 (40) | 340 (24) |
| simply_pet_joey | 3.440 (138) | 0 (0) | 0 (0) |

Số nhãn >=10 cm là số cửa sổ có giá trị tương lai cao, KHÔNG phải số đợt ngập độc lập. Các cửa sổ liên tiếp có tương quan mạnh.

**Không nên gộp ba trạm rồi bỏ qua cấu trúc này:** tired_red_monkey chỉ xuất hiện sau ranh giới train, simply_pet_joey không có validation/test, còn daily_new_falcon có rất ít nhãn cao ở các tập đánh giá. Không tự chia ngẫu nhiên để tạo kết quả đẹp.

## Baseline thăm dò (chưa phải đánh giá cuối)

Baseline persistence không cần train: dự đoán độ sâu tại t+10 phút bằng độ sâu tại t.

Với daily_new_falcon/test:

- MAE toàn bộ 165.330 cửa sổ: khoảng **0,00388 cm**.
- MAE riêng 15 cửa sổ có nhãn >=10 cm: khoảng **6,39 cm**.

Sự khác biệt lớn cho thấy MAE toàn tập ngày khô dễ tạo ấn tượng tốt giả. Hai con số đều chỉ là thăm dò trên proc, có thiên lệch chọn cửa sổ và chưa xác minh tính nhân quả bộ lọc. Không dùng chúng để tuyên bố hiệu quả dự báo tại trạm thật hoặc chọn siêu tham số theo test. Các mốc thời gian và chỉ số này hiện chỉ là khảo sát dữ liệu, cần giữ một tập đánh giá cuối chưa sử dụng trước khi tối ưu model thực sự.

## Đề xuất tiếp theo

1. Xác minh cách lọc proc và độ trễ của FloodNet; nếu không xác minh được, ghi rõ nghiên cứu hồi cứu hoặc xây bộ xử lý causal riêng từ dữ liệu phù hợp.
2. Mở rộng thống kê cửa sổ sang các trạm pluvial khác, chọn nhiều sự kiện và chu kỳ tương thích, không chỉ chọn theo số đoạn vượt ngưỡng.
3. Thiết kế riêng pipeline bước 5 phút cho trạm cũ hoặc lag theo thời gian với dung sai/độ mới rõ ràng; không đơn thuần nội suy thành hàng phút.
4. Đối soát nhóm sự kiện có nhãn cao và các cửa sổ ngay trước khi vượt ngưỡng, tránh tự xóa nước dâng nhanh.
5. Chốt lại tập train/validation/test và kiểm thử cuối trước khi huấn luyện, dùng baseline làm mốc, báo cáo riêng lúc có ngập và cảnh báo trước khởi phát.

## File và kiểm thử

- Mở `index.html` bằng trình duyệt để xem toàn bộ 12 biểu đồ.
- Chi tiết: `plot_cases.csv`, `window_counts.csv`, `preparation_summary.json`.
- Dữ liệu tạm: `provisional_windows/*.parquet`; cache: `local_cache/*.parquet`.
- Script: `prepare_data.py`; 3 kiểm thử tổng hợp trong `test_prepare_data.py` đã đạt.
- Không sửa CSV gốc. Không chạy API, không chỉnh frontend, không train XGBoost.

Nguồn: **FloodNet (New York University and The City University of New York).** Giữ đầu ra cục bộ; không phân phối dữ liệu gốc hoặc cache/cửa sổ khi chưa làm rõ quyền. Phân tích/biểu diễn phái sinh được phép: CC BY-NC-SA 4.0 cùng điều kiện bổ sung của Data Access License Agreement.
