# Sàng lọc 58 trạm pluvial với bước 5 phút

**Trạng thái: bộ thăm dò, chưa train model và chưa xác minh tính nhân quả bộ lọc proc.**

Đã quét 58 lần triển khai; 3 trạm đạt chính sách sàng lọc train; chọn 3 trạm core (tối đa 20, không ép đủ 10).

## Phân nhóm

| Nhóm | Số trạm |
|---|---:|
| no_train_windows | 32 |
| review_quality | 18 |
| limited_wet_evidence | 4 |
| candidate_core | 3 |
| insufficient_train_coverage | 1 |

candidate_core: đạt sàng lọc tự động; review_quality: cần đối soát; limited_wet_evidence: chưa đủ cửa sổ có nước đa dạng; insufficient_train_coverage: thiếu độ phủ train; no_train_windows: không có train theo mốc hiện tại. Không nhóm nào được coi là xác nhận ngập/lỗi hiện trường.

## Quy tắc chọn — chỉ dùng train

Train có >=1.000 cửa sổ, >=30 ngày có cửa sổ, >=20 nhãn >=10 cm, >=1.000 nhãn <10 cm, >=3 ngày có nhãn cao và >=3 nhóm nhãn cao cách nhau hơn 6 giờ.
Cần tỷ lệ cửa sổ có cờ raw/proc lệch >=10 cm không quá 10%; cờ độ dốc proc >25,4 cm/phút không quá 0,5%; bin proc thiếu trong giai đoạn train quan sát không quá 30%. Đây là ngưỡng thiết kế sàng lọc, chưa được chứng minh tối ưu, không được chỉnh theo test.
Thiếu bin không đồng nghĩa bản tin null: bao gồm khoảng không quan sát, chu kỳ gửi thưa và mất liên lạc. Không ép cân bằng bằng số ngày khô tổng hợp hoặc nội suy.

Xếp hạng các trạm đạt theo số nhóm nhãn cao 6h, số ngày có nhãn cao, rồi cờ độ dốc, cờ raw/proc, bin thiếu và ID. Xếp hạng này chưa tối ưu đa dạng không gian/lưu vực; không mặc định nhiều nhóm là nhiều trận mưa thật.

## Trạm core tạm thời

| ID | Trạm | Cửa sổ train | Nhãn >=10 cm | Ngày có nhãn cao | Nhóm 6h |
|---|---|---:|---:|---:|---:|
| daily_new_falcon | BK - Hoyt St/5th St | 208,388 | 737 | 23 | 19 |
| ugliest_cyan_elephant | SI - Loop Rd | 33,849 | 427 | 14 | 13 |
| widely_proud_lizard | BK - 9th St/Smith St (RETIRED) | 28,076 | 56 | 3 | 3 |

## Đối soát trực quan bổ sung

Giữ nguyên kết quả sàng lọc tự động; không điều chỉnh ngưỡng theo ảnh hoặc test. Các ghi chú sau chỉ cho trường hợp đỉnh TRAIN đã xem, không xác minh toàn bộ sự kiện của trạm.
- **daily_new_falcon** (needs_context, 2021-02-14 19:25 UTC): Mẫu thưa; có một đỉnh cao giữa các mức gần 18–20 cm. Raw/filt/proc trùng ở nhiều mẫu. Chỉ trùng nhau không xác nhận ngập thật; cần xem thời gian dài hơn và bối cảnh.
- **ugliest_cyan_elephant** (plausible_shape_not_verified, 2022-09-13 08:35 UTC): Có chuỗi tăng dần từ gần 0 lên khoảng 60 cm rồi rút dần; ba trường độ sâu gần trùng. Hình thái phù hợp để nghiên cứu chuỗi dâng/rút, nhưng chưa có xác minh độc lập.
- **widely_proud_lizard** (priority_review_possible_box, 2021-12-09 14:15 UTC): Từ gần 0 nhảy lên khoảng 98 cm, sau đó giữ khoảng 80 cm rồi về 0. Hình thái có thể tương tự box/vật cản; không đủ căn cứ để xác nhận lỗi hay ngập. Chưa nên tự động nhận mọi cửa sổ của trạm làm nhãn ngập đã kiểm định.

**Ba trạm core là ứng viên tạm thời, không phải ba trạm đã được xác nhận nhãn sạch.**

## Cửa sổ theo tập

| Phạm vi | Split | Cửa sổ | Nhãn >=10 cm |
|---|---|---:|---:|
| Tất cả trạm | train | 1,188,590 | 8,699 |
| Tất cả trạm | validation | 1,036,916 | 3,413 |
| Tất cả trạm | test | 682,848 | 4,935 |
| Core tạm thời | train | 270,313 | 1,220 |
| Core tạm thời | validation | 92,215 | 1,037 |
| Core tạm thời | test | 42,218 | 137 |

## Ý nghĩa thời gian và giới hạn

- Bin 5 phút lấy nguyên hàng quan trắc cuối, giữ null, không trộn ba cột từ ba thời điểm khác nhau; không nội suy/fill.
- Lịch sử gồm 3 bin t-10/t-5/t; support thực tế 15 phút vì mỗi bin rộng 5 phút. Độ mới số đo được lưu làm đặc trưng.
- Nhãn là bin ở t+10 phút, tức số đo cuối trong (t+5,t+10]; không đồng nhất với độ sâu tức thời đúng t+10.
- Train trước 01/01/2023, validation từ 01/01 đến trước 01/07/2023, test từ 01/07/2023; embargo 24 giờ mỗi bên ranh giới, bao gồm toàn support lịch sử và nhãn.
- Embargo 24 giờ là lựa chọn thận trọng để giảm liên hệ cùng đợt mưa/thoát nước qua ranh giới, không phải yêu cầu toán học của horizon 10 phút. Nó có thể làm mất cửa sổ ngập hiếm và chưa bảo đảm độc lập khí tượng.
- Thay mốc chia so với bộ ba trạm trước để có thêm lịch sử train; mốc này cố định trước chạy sàng lọc, không chọn bằng điểm dự báo.
- Nhóm trạm chỉ xuất hiện muộn được giữ riêng để khảo sát chuyển vị trí. Không xóa hoặc tự chia ngẫu nhiên cho đủ tập.
- Đã kiểm tra thêm: 32/32 trạm no_train_windows bắt đầu có số đo từ mốc train kết thúc trở đi. Không thể đưa chúng vào train trước 2023 mà vẫn giữ cách chia này. Không hạ ngưỡng tự động để ép đủ 10–20 trạm.
- Tỷ lệ thiếu bin là tiêu chí thận trọng cấp trạm; trạm bị xếp review_quality vẫn có thể có nhiều cửa sổ tốt. Không coi toàn bộ nhóm này là hỏng hoặc vĩnh viễn loại khỏi nghiên cứu.
- Năm 2023 đã được khảo sát trước đó: KHÔNG gọi test này là holdout xác nhận chưa từng xem. Cần một bộ kiểm định độc lập sau khi chốt toàn pipeline.
- Các cột review_* và metadata không tự đưa vào đặc trưng model. Không dùng nhãn tương lai làm đặc trưng.
- Các cột audit_* chứa metadata tương lai của nhãn: TUYỆT ĐỐI không train bằng chúng. Manifest có whitelist feature_columns. window_counts.csv ghi độ mới đầu vào và phân vị lead time thực tế.
- Các tập validation/test thuộc mùa khác nhau: khi đánh giá model cần tách thống kê theo mùa/sự kiện, không suy mọi chênh lệch là khả năng tổng quát.
- Chưa so sánh XGBoost nhóm core với tất cả trạm đủ điều kiện. Không thể kết luận nhóm nào tốt hơn khi chưa train/validation.
- Cần tiếp tục đối soát các trạm core và phương pháp lọc causal trước khi dùng dữ liệu vào đánh giá thời gian thực.

## Đầu ra

- `station_selection.csv`: danh sách 58 trạm, phân nhóm, lý do và thống kê train.
- `window_counts.csv`: số cửa sổ từng trạm/split; validation/test chỉ dùng mô tả, không dùng xếp hạng.
- `selection_manifest.json`: quy tắc, trạm core, nguồn và giới hạn.
- `index.html`, `plots/`: biểu đồ kiểm tra chọn từ TRAIN, chưa xác minh thủ công tất cả.
- `provisional_windows/`: cửa sổ tất cả trạm/split, kể cả file rỗng, chưa phải bộ train được duyệt.
- `local_cache/`: dữ liệu trích xuất cục bộ theo tháng/trạm; fingerprint gồm kích thước, mtime và schema cache.

Nguồn: FloodNet (New York University and The City University of New York). Không sửa CSV gốc, không đưa dữ liệu lên dịch vụ bên ngoài. Giữ cache/cửa sổ cục bộ. Phân tích/biểu diễn phái sinh được phép tuân CC BY-NC-SA 4.0 và Data Access License Agreement; cần rà quyền trước công khai/bàn giao.
