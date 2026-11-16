# Kiểm kê chất lượng và ứng viên ngập FloodNet

## Kết luận

Dữ liệu đủ quy mô để bắt đầu thử nghiệm baseline và XGBoost. Chưa chứng minh model dự báo tốt hoặc áp dụng được tại Việt Nam. Chưa train model.

## Phạm vi quét

- 36 CSV; 4,459,382,121 byte; 17,154,579 bản ghi; 88 deployment_id.
- Các file từ 10/2020 đến 10/2023; thiếu file 09/2023. Không khẳng định mỗi trạm liên tục đủ ba năm.
- Loại triển khai sau chuẩn hóa khoảng trắng: {'pluvial': 58, 'coastal': 30}.
- Thời gian không hợp lệ: 0; đảo thời gian trong từng trạm/file: 0.
- Trùng thời điểm: 0; trùng có xung đột: 0.

## Chất lượng dữ liệu

- Thiếu/không hữu hạn depth_proc_mm: 129,891 (0.757% tổng số hàng).
- Khoảng giữa hai mẫu liên tiếp trên 10 phút: 111,274. Đây là số khoảng, không phải số hàng mất hoặc thời lượng mất.
- depth_raw_mm âm: 10,036,077 (58.504%). Không tự coi là lỗi: mốc nền và nhiễu có thể tạo độ sâu thô âm.
- Chênh lệch tuyệt đối raw/proc >=100 mm: 1,014,888 hàng. Không chứng minh bộ lọc sai.
- Raw >=100 mm nhưng proc <10 mm: 212,699 hàng; cần xem xét các cửa sổ liên quan trước chọn đặc trưng.
- Chuyển tiếp proc có độ dốc tuyệt đối >254 mm/phút trong khoảng liên tục: 519; cờ đối soát, không tự xóa.
- Không có kiểm định ngoại cảnh; chưa xác minh bộ lọc archive có sử dụng dữ liệu tương lai.

## Đợt ứng viên có duy trì

Điều kiện: >=3 mẫu khác thời điểm, kéo dài >=2 phút; tách khi xuống dưới ngưỡng, mẫu proc thiếu hoặc khoảng cách >max(3 phút, 3 lần chu kỳ trung vị tháng).

| Loại | >=1 cm | >=10 cm | >=20 cm |
|---|---:|---:|---:|
| coastal | 7,047 | 1,050 | 266 |
| pluvial | 10,131 | 2,829 | 1,432 |

Ba mức lồng nhau, không cộng lại thành tổng sự kiện. Một đợt ngập thật có thể bị chia thành nhiều đoạn do nhiễu, mất mẫu hoặc dao động quanh ngưỡng.
Các đoạn pluvial >=10 cm có duy trì xuất hiện theo năm bắt đầu: {'2020': 4, '2021': 83, '2022': 1134, '2023': 1608}.
Nhóm theo thời gian cách nhau <=6 giờ tạo 274 cụm; KHÔNG phải số trận mưa độc lập đã xác minh.

## Các trạm cần xem trước

Xếp theo số đoạn >=10 cm có duy trì, không phải xếp hạng chất lượng. Mẫu thiếu chỉ tính trên các hàng có mặt, không bao gồm mọi thời gian offline.

| deployment_id | Trạm | Số hàng | Chu kỳ tháng trung vị (s) | Thiếu proc | Đoạn >=10 cm | Max proc (cm) |
|---|---|---:|---:|---:|---:|---:|
| tired_red_monkey | Q - Brinkerhoff Ave/157th St | 207,340 | 63.46 | 1.643% | 1139 | 79.8 |
| simply_pet_joey | M - Waverly Pl/6th Av | 78,764 | 63.13 | 6.612% | 493 | 156.0 |
| duly_star_bee | M - Morningside Av/125th St (RETIRED) | 482,937 | 63.13 | 0.002% | 354 | 95.1 |
| weekly_fair_dove | BX - Colgate Av/Westchester Av | 535,828 | 63.1 | 0.818% | 259 | 104.0 |
| future_aqua_wildcat | SI - Lyon Pl/Devens St | 207,518 | 126.2 | 0.049% | 130 | 130.9 |
| daily_new_falcon | BK - Hoyt St/5th St | 1,029,482 | 63.0 | 0.004% | 115 | 48.6 |
| ugliest_cyan_elephant | SI - Loop Rd | 215,638 | 182.99 | 0.147% | 82 | 118.6 |
| jolly_tender_squid | BK - Henry St/Mill St | 462,381 | 63.13 | 0.000% | 71 | 13.9 |
| weekly_poetic_guinea | BK - Carroll St/4th Av | 313,344 | 126.11 | 0.000% | 50 | 95.8 |
| blue_eyed_tiger | BK - Lee Ave/Middleton St | 203,490 | 63.34 | 0.251% | 35 | 12.7 |
| early_still_frog | BK - Lorraine St/Henry St | 292,679 | 63.04 | 0.000% | 22 | 124.2 |
| mean_flying_fish | BK - Wallabout St/Throop Ave | 177,385 | 63.45 | 0.372% | 16 | 136.2 |

## Cảnh báo khi lựa chọn trạm

- Bốn trạm đứng đầu chiếm 2,245/2,829 đoạn pluvial >=10 cm (79.357%). Số lượng lớn tập trung không chứng minh chất lượng tốt; cần đối soát noise và độ phân mảnh.
- daily_new_falcon là ứng viên đáng xem trước: chuỗi dài, 115 đoạn >=10 cm có duy trì, chỉ 40 hàng thiếu proc trên 1.029.482 hàng, không có chuyển tiếp proc vượt 254 mm/phút theo phép quét. Đây là lựa chọn sơ bộ, không xác nhận mọi đoạn là ngập thật.
- Các trạm như simply_pet_joey và future_aqua_wildcat có nhiều đoạn >=10 cm đồng thời >=20 cm, có các đỉnh lớn và cờ độ dốc. Cần xem đồ thị, không tự loại hoặc coi tất cả là sự kiện ngập.

## Bước tiếp theo trước train

1. Xem đồ thị các cửa sổ ngập và raw/filt/proc ở nhóm pluvial; rà các đỉnh và bước nhảy, kể cả trạm có nhiều đoạn.
2. Chọn nhóm trạm có chu kỳ tương thích. Không nội suy từ 5 phút thành 1 phút rồi coi là quan trắc thật.
3. Xác minh bộ lọc causal/độ trễ; chỉ dùng đầu vào thực sự sẵn có tại thời điểm dự báo. Nhãn đã đối soát có thể là sản phẩm hậu kỳ.
4. Đếm số cửa sổ 10 phút hợp lệ và nhãn tương lai 10 phút sau khi xử lý khoảng thiếu; audit này chưa đếm cửa sổ train.
5. Chia thời gian/sự kiện và giữ tập test; purge cửa sổ chồng qua ranh giới; nhóm nhiều trạm cùng sự kiện để tránh rò rỉ.
6. So sánh baseline và XGBoost, riêng trên lúc có ngập và lúc chưa ngập trước khởi phát; không chỉ báo cáo MAE toàn bộ ngày khô.

## Đầu ra và giấy phép

Các bảng chi tiết nằm cạnh báo cáo: summary.json, files.csv, stations.csv, station_month_quality.csv, candidate_episodes.csv, temporal_clusters.csv. Không sửa CSV nguồn.
Nguồn: FloodNet (New York University and The City University of New York). Chỉ dùng phi thương mại theo thỏa thuận được cung cấp. Không phân phối dữ liệu gốc; cần rà quyền trước công khai các đầu ra dẫn xuất. Chưa xác minh ngày tải/thỏa thuận tương ứng của từng file; người tải cần bổ sung thông tin này.
Phân tích/biểu diễn phái sinh được phép: CC BY-NC-SA 4.0, kèm các điều kiện bổ sung của Data Access License Agreement.
