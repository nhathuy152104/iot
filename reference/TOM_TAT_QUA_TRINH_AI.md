# Tóm tắt quá trình chuẩn bị dữ liệu và train AI dự báo ngập

**Cập nhật:** 11/11/2026.  
**Thư mục làm việc:** `D:\CODE\IOT\IOT training`.  
**Trạng thái:** đã có bản đồ phát lại độ sâu dự báo tại `/forecast`, phân tích lỗi quanh ngưỡng và một thử nghiệm objective MAE riêng, ngoài các bước kiểm kê/train/đánh giá A–B. **Giữ model raw ba mã cũ cho demo; thử nghiệm mới giảm MAE toàn tập nhưng chưa cải thiện toàn diện. Chưa chạy AI trên ESP32 hoặc đưa vào cảnh báo/định tuyến thật.** Xem cập nhật ở mục 13–18.

## 1. Mục tiêu và hướng tiếp cận

- Dự án chính là hệ thống ESP32 → MQTT → backend → bản đồ/cảnh báo ngập.
- Đã có dữ liệu ESP32 thật, nhưng lịch sử và số đợt ngập chưa đủ để tự train/kiểm định model đáng tin cậy. Buổi demo chỉ khoảng 5–10 phút.
- Chọn FloodNet NYC dạng chuỗi cảm biến độ sâu để nghiên cứu pipeline trước; không nhầm với bộ ảnh cùng tên.
- Bài toán đầu tiên: dùng lịch sử độ sâu khoảng 10 phút để dự báo độ sâu ở bin tương lai t+10 phút.
- Bản hiện tại không dùng tọa độ, đoạn đường, mưa hay địa hình làm đặc trưng. Chủ yếu dùng độ sâu trễ, tốc độ dâng/rút, thống kê lịch sử và độ mới số đo.
- Dùng XGBoost chạy CPU, so sánh với các baseline đơn giản; chưa chọn LSTM, camera hoặc chatbot.
- Đã triển khai trang phát lại lịch sử cục bộ tại `/forecast` ngày 11/11/2026, có thời gian tăng tốc và các dự báo tính từ lịch sử có sẵn. Tách khỏi ESP32 và bản đồ live; không nối lịch sử NYC với số đo ESP32 thành một chuỗi giả. Xem mục 18.

## 2. Dữ liệu đã kiểm kê

### 2.1. Kiểm tra ban đầu

Đã đọc file `D:\Downloads\floodnet-flood-sensor-sep-2021.csv`: đúng dạng dữ liệu cần dùng, nhưng phần lớn là số đo không ngập và ít ngày có mẫu ngập lớn. Một tháng đủ để thử pipeline, không đủ để kết luận khả năng dự báo.

### 2.2. Quét toàn bộ dữ liệu trong thư mục hiện tại

| Hạng mục | Kết quả |
|---|---:|
| CSV | 36 file |
| Dung lượng | 4.459.382.121 byte, khoảng 4,46 GB |
| Phạm vi theo file | 10/2020–10/2023 |
| File còn thiếu trong khoảng trên | 09/2023 |
| Tổng bản ghi | 17.154.579 |
| Lần triển khai cảm biến (`deployment_id`) | 88 |
| Ngập do mưa (`pluvial`) | 58 |
| Ven biển/thủy triều (`coastal`) | 30 |
| Thiếu/không hữu hạn `depth_proc_mm` | 129.891 hàng, khoảng 0,76% |
| Khoảng cách giữa hai mẫu >10 phút | 111.274 khoảng |
| Thời gian không hợp lệ | 0 |

Đã chuẩn hóa nhãn loại triển khai có khoảng trắng thừa trong thống kê dẫn xuất. Một deployment không nhất thiết hoạt động liên tục suốt ba năm.

### 2.3. Đếm đợt ứng viên

Định nghĩa ban đầu: đoạn liên tiếp vượt ngưỡng, có ít nhất 3 mẫu khác thời điểm và kéo dài từ 2 phút. Tách đoạn khi xuống dưới ngưỡng, thiếu proc hoặc khoảng cách giữa mẫu vượt `max(3 phút, 3 × chu kỳ trung vị tháng)`.

| Ngưỡng tại trạm pluvial | Số đoạn có duy trì |
|---|---:|
| ≥1 cm | 10.131 |
| ≥10 cm | 2.829 |
| ≥20 cm | 1.432 |

Ba mức lồng nhau, không cộng lại. Nhóm các đoạn pluvial ≥10 cm theo khoảng cách thời gian tối đa 6 giờ tạo 274 cụm. **Các đoạn/cụm này chưa phải trận ngập hoặc trận mưa độc lập đã xác minh.**

Các vấn đề đã nhận diện:

- Nhiều hàng không đồng nghĩa nhiều sự kiện hữu ích; phần lớn dữ liệu là mức thấp/0.
- Có trạm gửi khoảng 1 phút, trạm cũ gửi khoảng 5 phút hoặc thưa hơn.
- Raw âm không tự động là lỗi; có thể liên quan mốc nền, hiệu chuẩn và nhiễu.
- Có 212.699 hàng raw ≥10 cm nhưng proc <1 cm; cần hiểu bộ lọc, không coi raw mặc nhiên là đúng.
- Có blip, box, dao động phản xạ, khoảng mất mẫu và khác biệt raw/proc. Không tự xóa mọi nước dâng nhanh vì có thể là ngập thật.

**Chi tiết:** [Báo cáo kiểm kê](audit/BAO_CAO_KIEM_KE.md), [thống kê tổng](audit/summary.json).

## 3. Thử pipeline 1 phút và đối soát ban đầu

Đã chọn ba trạm có đặc điểm khác nhau để khảo sát:

- `daily_new_falcon`: lịch sử dài, tương đối ít cờ theo audit.
- `tired_red_monkey`: nhiều đoạn vượt ngưỡng, cần xem sự phân mảnh/nhiễu.
- `simply_pet_joey`: lịch sử ngắn, nhiều mức cao biến động lớn.

Đã tạo 12 biểu đồ raw/filt/proc, trực tiếp xem 4 ca, tạo cửa sổ lịch sử 10 phút với đủ 11 bin một phút và nhãn t+10 phút. Không nội suy/fill.

Pipeline 1 phút quá chặt với trạm gửi thưa: `daily_new_falcon` có nhiều cửa sổ ngày khô nhưng chỉ 15 nhãn test ≥10 cm. Hai trạm còn lại không có đủ train/validation/test theo cách chia ban đầu.

**Kết luận:** giữ bộ này làm đối chiếu, chuyển sang bước 5 phút để phù hợp chu kỳ đo. Không trực tiếp so các tổng số cửa sổ 1 phút/5 phút như một thí nghiệm chỉ thay một yếu tố: phạm vi trạm, yêu cầu cửa sổ và mốc chia tập cũng đã thay đổi.

**Chi tiết:** [Báo cáo đối soát ban đầu](analysis/BAO_CAO_DOI_SOAT.md), [biểu đồ](analysis/index.html).

## 4. Mở rộng toàn bộ 58 trạm ở bước 5 phút

### 4.1. Quy tắc chuẩn bị

- Mỗi bin UTC là `(t-5 phút,t]`, lấy **nguyên hàng quan trắc cuối**, giữ cả null. Không ghép các cột từ những thời điểm khác nhau.
- Không nội suy, không forward/backward fill.
- Đầu vào cần ba bin proc không âm tại t-10, t-5 và t. Khoảng support thực tế rộng 15 phút vì mỗi bin có độ rộng 5 phút.
- Nhãn là số đo cuối trong bin `(t+5 phút,t+10 phút]`, **không phải độ sâu tức thời đúng t+10** và không phải mức cực đại của cả khoảng tương lai.
- Lưu độ mới đầu vào và thời điểm thật của nhãn. Metadata tương lai `audit_*`, cờ `review_*`, nhãn và ID trạm không được đưa vào đặc trưng model.

### 4.2. Chia tập cố định

| Tập | Khoảng thời gian quy ước |
|---|---|
| Train | Trước 01/01/2023 UTC |
| Validation | Từ 01/01/2023 đến trước 01/07/2023 UTC |
| Test | Từ 01/07/2023 UTC |

Loại cửa sổ chồng ranh giới theo toàn bộ support và thêm embargo 24 giờ mỗi bên. Khoảng đệm này là quy ước thận trọng, không chứng minh các sự kiện khí tượng hoàn toàn độc lập.

### 4.3. Chọn trạm bằng phần train

Mục tiêu ban đầu là tìm khoảng 10–20 trạm, nhưng không ép đủ số lượng. Chỉ dùng train để xếp nhóm; không dùng điểm model hoặc thống kê test để chọn trạm.

Tiêu chí sàng lọc gồm: ≥1.000 cửa sổ, ≥30 ngày có cửa sổ, ≥20 nhãn ≥10 cm, ≥1.000 nhãn <10 cm, ≥3 ngày có nhãn cao và ≥3 nhóm nhãn cao cách nhau hơn 6 giờ. Kèm giới hạn tỷ lệ bin thiếu và cờ raw/proc/độ dốc. Đây là quy tắc thiết kế thăm dò, không phải chứng nhận nhãn sạch.

| Nhóm | Số trạm |
|---|---:|
| Đạt sàng lọc tự động (`candidate_core`) | 3 |
| Cần đối soát chất lượng | 18 |
| Ít bằng chứng ngập đa dạng trong train | 4 |
| Thiếu độ phủ train | 1 |
| Không có cửa sổ train theo mốc hiện tại | 32 |

Cả 32 trạm cuối bắt đầu có quan trắc từ 2023; không có nghĩa dữ liệu hỏng. Không hạ ngưỡng tự động để đạt 10–20 trạm.

Ba trạm core tạm thời:

1. `daily_new_falcon` — BK, Hoyt St/5th St.
2. `ugliest_cyan_elephant` — SI, Loop Rd.
3. `widely_proud_lizard` — BK, 9th St/Smith St, đã ngừng triển khai theo tên nguồn.

| Phạm vi | Train | Validation | Test |
|---|---:|---:|---:|
| Toàn bộ 58 trạm | 1.188.590 | 1.036.916 | 682.848 |
| Ba core tạm thời | 270.313 | 92.215 | 42.218 |
| Nhãn ≥10 cm trong core | 1.220 | 1.037 | 137 |

Tổng 2.908.354 cửa sổ trong 174 file Parquet đã kiểm tra cấu trúc thành công. Đã xem ca đỉnh train của cả ba core; `widely_proud_lizard` có hình thái nghi box nên vẫn cần đối soát, không coi nhóm core là ground truth sạch.

**Chi tiết:** [Báo cáo 5 phút](analysis/five_minute/BAO_CAO_5_PHUT.md), [danh sách trạm](analysis/five_minute/station_selection.csv), [manifest](analysis/five_minute/selection_manifest.json), [biểu đồ](analysis/five_minute/index.html).

## 5. Đối soát thêm nhóm 18 trạm

Đã phân biệt null trên hàng có mặt với bin không quan sát, tìm các đoạn liên tục và kiểm tra hình thái các đoạn cao. Chỉ dùng train, không sửa cửa sổ hoặc nhóm core gốc.

- Tạo 75 biểu đồ, xem ca đỉnh của cả 18 trạm qua contact sheet và 8 ca liên tục bổ sung.
- Kiểm tra 682.662 cửa sổ train.
- 625.790 cửa sổ thuộc nhóm ít cờ theo quy tắc đối soát.
- 56.872 cửa sổ được đánh dấu xem riêng, **không xóa**.
- Có 7 trạm đủ số lượng/đa dạng cửa sổ ít cờ để nghiên cứu bổ sung; 11 trạm có thể làm dữ liệu nền hoặc còn ít bằng chứng sự kiện đa dạng.

Bảy trạm đề xuất xem xét theo đoạn:

`daily_gentle_beetle`, `evenly_divine_dingo`, `overly_heroic_squid`, `simply_pet_joey`, `tired_red_monkey`, `weekly_fair_dove`, `weekly_poetic_guinea`.

**Chưa tự động gộp cả bảy vào core.** Một số ca vẫn giữ mức cao lâu, chuyển mức đột ngột hoặc dao động mạnh dù raw/proc trùng. Ít cờ không đồng nghĩa nhãn đúng; một ca tăng/rút đẹp cũng không xác nhận cả trạm.

Mask đối soát có dùng ngữ cảnh cuối sự kiện và nhãn tương lai: chỉ dùng phân tích hồi cứu/độ nhạy, **không dùng làm feature hoặc bộ lọc vận hành online**. Không chỉ chấm điểm model trên nhóm ít cờ rồi bỏ các ca khó. Chưa kiểm tra thời tiết, hiện trường hay nhãn đo độc lập.

**Chi tiết:** [Báo cáo 18 trạm](analysis/five_minute/review_18/BAO_CAO_DOI_SOAT_18.md), [ghi chú trực quan](analysis/five_minute/review_18/visual_notes.json), [biểu đồ](analysis/five_minute/review_18/index.html).

## 6. Môi trường và công cụ đã cài

Môi trường riêng: `analysis\.venv`, không thay môi trường backend.

| Công cụ | Phiên bản ghi nhận |
|---|---|
| Python | 3.15.2 |
| pandas | 3.1.1 |
| NumPy | 2.5.4 |
| Matplotlib | 3.11.2 |
| PyArrow | 26.0.1 |
| XGBoost | 3.4.1 |
| scikit-learn | 1.9.1 |

Chạy được trên CPU laptop, tối đa 4 luồng ở lượt train đầu. Không cài CUDA, không cần GPU hoặc Docker cho pipeline này. Phiên bản trực tiếp được ghi trong [requirements.txt](analysis/requirements.txt); thông tin runtime nằm trong cấu hình mỗi lượt train.

## 7. Lượt train đầu tiên

**Run:** `analysis/training_runs/20261110T033843_342867Z`.

### 7.1. Thiết kế thí nghiệm

- Train bằng toàn bộ cửa sổ gốc của ba core, không dùng mask offline để loại ca khó.
- Validation/test thực tế chỉ có hai trạm còn dữ liệu; `widely_proud_lizard` chỉ đóng góp train.
- Hai baseline: persistence (giữ độ sâu hiện tại) và ngoại suy tốc độ 5 phút vừa qua thêm 10 phút. Dự đoán âm được giới hạn về 0.
- Bốn cấu hình XGBoost: depth3/250 cây hoặc depth4/400 cây, mỗi loại có bản trọng số thường và bản nhân20 cho nhãn train ≥10 cm.
- Learning rate 0,05; seed42. Không dùng ID trạm, metadata tương lai hoặc nhãn làm feature.
- Chọn model bằng MAE cân bằng trên validation: trung bình MAE của nhóm nhãn ≥10 cm và <10 cm.
- Chốt model trước nạp test, không refit bằng validation và không điều chỉnh theo test.
- Chọn `d3_weight20`, là tốt nhất **trong bốn cấu hình XGBoost**, không phải tốt nhất so với baseline.

### 7.2. Kết quả

| Chỉ số test | Persistence | Xu hướng tuyến tính | XGBoost |
|---|---:|---:|---:|
| MAE toàn tập (cm) | **0,0913** | 0,1436 | 0,1848 |
| MAE khi nhãn ≥10 cm (cm) | **5,51** | 9,07 | 14,84 |
| MAE cân bằng (cm) | **2,79** | 4,59 | 7,49 |
| Recall nhãn ≥10 cm | 68,6% | 70,8% | **75,9%** |
| Precision nhãn ≥10 cm | **68,6%** | 42,4% | 40,3% |
| Cửa sổ báo giả | **43** | 132 | 154 |
| Cửa sổ bỏ sót | 43 | 40 | **33** |
| Phát hiện cửa sổ hiện tại <10, tương lai ≥10 cm | 0/43 | 9/43 | 10/43 |

Trên validation, MAE cân bằng của persistence là **0,4851 cm**, tốt hơn **0,5130 cm** của XGBoost được chọn.

**Kết luận:** train thành công về kỹ thuật nhưng chưa chứng minh AI hơn baseline. XGBoost bỏ sót ít hơn, đổi lại báo giả và sai số lớn hơn. Chưa đưa model vào cảnh báo thực tế.

Test chỉ có 137 cửa sổ nhãn ≥10 cm, trong đó 131 thuộc `ugliest_cyan_elephant`. Các hàng tương quan, không phải 137 trận ngập độc lập. 43 cửa sổ chuyển ngưỡng cũng không phải 43 lần khởi phát độc lập hoặc phép đo lead time cảnh báo.

Đã xem hai biểu đồ test sau chốt model. Có ca mức cao trên100cm tại `ugliest_cyan_elephant` mà XGBoost dự đoán thấp đáng kể; chưa xác nhận đó là ngập thật hoặc nguyên nhân model sai. Không xóa ca này để tăng điểm.

### 7.3. Artifacts

Trong thư mục run:

- [KET_LUAN.md](analysis/training_runs/20261110T033843_342867Z/KET_LUAN.md): kết luận chính.
- [BAO_CAO_TRAIN.md](analysis/training_runs/20261110T033843_342867Z/BAO_CAO_TRAIN.md): báo cáo thí nghiệm.
- `xgboost_selected.ubj`: model lưu cục bộ, chưa được duyệt deployment.
- `config_before_fit.json`, `selection_before_test.json`: cấu hình trước train và lựa chọn trước test.
- `metrics.csv`, `test_breakdown.csv`, `summary.json`: chỉ số tổng và theo trạm/quý.
- `test_predictions_local.parquet`: dự đoán dẫn xuất, không công khai.
- `input_provenance.json`: đường dẫn, số hàng và checksum nguồn đã dùng.
- `artifact_validation.json`: xác nhận tái lập dự đoán/chỉ số thành công.

## 8. Kiểm thử và kiểm tra đã thực hiện

Các kết quả PASS được ghi nhận ở từng giai đoạn, không phải một lệnh test tổng duy nhất:

| Phần | Kết quả |
|---|---|
| Audit ban đầu | 2 kiểm thử tổng hợp đạt |
| Chuẩn bị 1 phút | 3 kiểm thử đạt |
| Pipeline 5 phút | 9 kiểm thử đạt |
| Đối soát 18 trạm + pipeline 5 phút | 16 kiểm thử đạt, gồm lại 9 kiểm thử trên |
| Baseline/chỉ số/chọn model | 4 kiểm thử đạt |
| File cửa sổ 5 phút | 174 file, 2.908.354 hàng: PASS |
| Mask đối soát | 18 trạm, 682.662 hàng train: PASS |
| Artifacts train | Checksum7 file đầu vào không rỗng, nạp model và dự đoán lại 42.218 hàng test: PASS |

Các kiểm tra kỹ thuật không xác minh nhãn ngập thật, tính nhân quả của bộ lọc hoặc độ an toàn vận hành.

## 9. Những giới hạn chưa giải quyết

1. **Bộ lọc `depth_proc_mm` chưa được xác minh có dùng dữ liệu tương lai không.** Nếu có, đánh giá hồi cứu có thể tốt hơn khả năng chạy trực tiếp. Tổng hợp bin causal không sửa được bộ lọc nguồn không causal.
2. Chưa có kiểm định hiện trường/nhãn độ sâu độc lập; một số mức cao có thể do vật cản, phản xạ hoặc hiệu chuẩn.
3. Dữ liệu 2023 đã được xem thống kê và biểu đồ trước; test hiện tại không phải holdout xác nhận hoàn toàn chưa từng xem.
4. Dữ liệu ngập mất cân bằng, ít sự kiện đa dạng và khác mùa/trạm giữa các tập. MAE toàn tập thấp dễ bị ngày khô chi phối.
5. Một điểm đo không mô tả độ sâu lớn nhất cả con đường. Ngưỡng 10/20 cm chỉ là quy ước nghiên cứu, không bảo đảm có thể đi qua an toàn.
6. Kết quả trên NYC không chứng minh model dùng tốt ở địa bàn ESP32 của dự án. Chưa kiểm định chuyển miền.

## 10. Việc chưa làm và hướng tiếp theo

**Chưa làm:** train bổ sung nhóm bảy trạm, xác minh bộ lọc nguồn, đối soát thời tiết/hiện trường, tích hợp model vào API/frontend, chạy model trên dữ liệu ESP32 hoặc triển khai cảnh báo thực tế. **Phân tích lỗi và thí nghiệm delta đã làm sau bản tóm tắt ban đầu; xem mục 13.**

Hướng tiếp theo đã thảo luận nhưng **chưa thực hiện**:

1. Chẩn đoán những đoạn gây sai số lớn: nhãn có đáng tin, train có diễn biến tương tự, sai ở khởi phát/giữ mức cao/rút nước hay không. Không loại dữ liệu chỉ vì model sai.
2. Tìm tài liệu/code hoặc hỏi FloodNet về tính nhân quả và độ trễ bộ lọc.
3. Thử XGBoost dự báo **mức thay đổi** thay vì độ sâu tuyệt đối: `độ sâu tương lai = độ sâu hiện tại + mức thay đổi dự báo`. Đây là ý tưởng cần kiểm nghiệm, chưa bảo đảm cải thiện.
4. Bổ sung các đoạn phù hợp từ trạm đã đối soát trong thí nghiệm riêng; báo cáo độ nhạy khi có/không có nhóm nghi vấn, không chỉ lấy ca đẹp.
5. Định nghĩa thêm đánh giá theo sự kiện/khởi phát, số lần báo giả và mức bỏ sót chấp nhận được. Chỉ tối ưu bằng train/validation; cần kiểm định độc lập cho khẳng định cuối cùng.

## 11. Các lệnh chạy lại chính

Các lệnh dưới đây là để tham khảo, **không tự chạy khi đọc tài liệu**. Chạy lại có thể tốn thời gian và cập nhật các báo cáo dẫn xuất; riêng train tạo thư mục run mới.

```powershell
# Kiểm kê CSV nguồn
python "D:\CODE\IOT\IOT training\audit\audit_floodnet.py"
python "D:\CODE\IOT\IOT training\audit\summarize_audit.py"

# Chuẩn bị/sàng lọc 5 phút và kiểm tra file đầu ra
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\five_minute\pipeline.py"
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\five_minute\validate_outputs.py"

# Đối soát 18 trạm
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\five_minute\review_18.py"
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\five_minute\validate_review_18.py"

# Train thử nghiệm; mỗi lần tạo một run mới
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\train_baseline_xgboost.py"

# Kiểm tra lại artifacts của lượt train đã có, không refit
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\run_relocated.py" "D:\CODE\IOT\IOT training\analysis\validate_training.py" "D:\CODE\IOT\IOT training\analysis\training_runs\20261110T033843_342867Z"
```

## 12. Nguồn dữ liệu và quyền sử dụng

**Lưu ý di chuyển thư mục (11/11/2026):** provenance JSON cũ được giữ nguyên đường dẫn lịch sử để không thay checksum. Khi chạy các script đọc provenance cũ, dùng `analysis/run_relocated.py` như các lệnh trong `analysis/TRAINING_README.md`. Wrapper ánh xạ đường dẫn JSON trong bộ nhớ, không bỏ kiểm tra checksum; giữ hành vi ghi file/train của script gốc. Không tự chạy các lệnh tham khảo khi chỉ đọc tài liệu.

**Nguồn:** FloodNet (New York University and The City University of New York).

Theo Data Access License Agreement người dùng cung cấp:

- Chỉ sử dụng phi thương mại theo thỏa thuận; dữ liệu công khai không có nghĩa được dùng/phân phối vô điều kiện.
- Không chia sẻ/tái đăng dữ liệu gốc, kể cả một phần, cho bên thứ ba khi chưa có chấp thuận bằng văn bản.
- Phân tích/biểu diễn phái sinh được phép phải tuân CC BY-NC-SA 4.0 cùng các điều khoản bổ sung, trích nguồn và không gây hiểu nhầm.
- Cần lưu ngày tải và bản thỏa thuận áp dụng; thông tin này của từng CSV chưa được xác minh đầy đủ.
- Giữ CSV, cache, cửa sổ, mask và dự đoán cục bộ. Làm rõ quyền trước công khai/bàn giao trọng số hoặc các artifact dẫn xuất.

Trong các bước xử lý đã làm, CSV nguồn được giữ nguyên; không đưa quan trắc lên dịch vụ bên ngoài, không chỉnh backend/frontend hoặc tự triển khai model. Trợ lý phụ từng được dùng chỉ để góp ý phương pháp, không đọc hoặc nhận dữ liệu gốc.

## 13. Cập nhật: chẩn đoán lỗi và thí nghiệm delta

**Ngày:** 10/11/2026. Không cài thêm thư viện, không chỉnh CSV/cửa sổ nguồn hoặc artifacts lượt train đầu.

### 13.1. Chẩn đoán model cũ

- Nạp lại model cũ và tái lập mọi chỉ số validation/test từ dữ liệu có checksum khớp: PASS.
- Cực đại nhãn train toàn core 97,8 cm; riêng `ugliest_cyan_elephant` 62,8 cm; test có mức 118,6 cm.
- 27 cửa sổ test có nhãn vượt cực đại train chiếm khoảng 72,8% tổng sai số tuyệt đối trên nhãn ≥10 cm của XGBoost cũ. Đây là phân rã hồi cứu, không loại các cửa sổ đó khỏi điểm chính.
- Trực tiếp xem 6 biểu đồ raw/filt/proc và dự đoán. Một số ca có chuyển mức đột ngột/mức cao kéo dài; chưa xác nhận ngập thật hoặc lỗi bằng dữ liệu độc lập. Không tự sửa/xóa nhãn.
- Weight20 làm giảm bỏ sót nhưng tăng báo giả trong cả hai cặp cấu hình cũ.

Đầu ra: `analysis/diagnostic_runs/20261110T040149_598786Z`, gồm `BAO_CAO_CHAN_DOAN.md`, `NHAN_XET_TRUC_QUAN.md`, bảng phân rã/đoạn ứng viên và 6 biểu đồ.

### 13.2. Thí nghiệm dự báo mức thay đổi

**Run:** `analysis/training_runs/20261110T040220_424848Z_delta`.

- Giữ nguyên 270.313 cửa sổ train, 92.215 validation, ba trạm core, 12 feature, bốn cấu hình, seed và trọng số mẫu.
- Chỉ đổi đích học: `future_depth - current_depth`; dự báo bằng `max(0, current_depth + predicted_delta)`.
- Không dùng mask offline, không thêm bảy trạm, không refit bằng validation. Trọng số vẫn dựa vào độ sâu tương lai ≥10 cm, không theo delta.
- **Không nạp/dự đoán test trong lượt delta.** Test cũ chỉ phục vụ chẩn đoán model cũ.
- Chọn `delta_d3_weight20` bằng MAE cân bằng trên validation; baseline cũng được phép thắng.

| Chỉ số validation | Persistence | XGBoost cũ | Delta XGBoost |
|---|---:|---:|---:|
| MAE cân bằng (cm) | 0,4851 | 0,5130 | **0,4725** |
| MAE nhãn ≥10cm (cm) | 0,9104 | 0,8994 | **0,8450** |
| MAE toàn tập (cm) | **0,0693** | 0,1352 | 0,1083 |
| Cửa sổ báo giả | **78** | 336 | 324 |
| Cửa sổ bỏ sót | 79 | **59** | 62 |

Delta giảm MAE cân bằng khoảng **2,6%** so với persistence trên validation, nhưng báo giả vẫn khoảng **4,15 lần**. Chưa thể kết luận AI hơn toàn diện hoặc dùng được để cảnh báo thật; validation đã được dùng nhiều lần, chưa có xác nhận độc lập cho model mới.

### 13.3. Kiểm tra và hướng tiếp theo

- 12 kiểm thử tổng hợp mới; cùng bộ test gốc trong `analysis/` có 19 kiểm thử đạt.
- Bộ test pipeline 5 phút/đối soát 18 trạm có thêm 16 kiểm thử đạt; tổng hai nhóm không trùng là 35 kiểm thử. Phân rã chẩn đoán theo pha/phạm vi train khớp tổng; 7 file nguồn, 3 cache và 6 biểu đồ được kiểm tra: PASS.
- Nạp lại bốn model delta, model cũ và hai baseline; tái lập 92.215 dự đoán validation cho mỗi phương pháp, chỉ số JSON/CSV và quyết định chọn model: PASS.
- 5 file train/validation khớp checksum/số hàng lượt gốc. Kiểm tra kỹ thuật không xác minh nhãn hoặc causality.
- Tiếp theo: định nghĩa chỉ số cảnh báo theo sự kiện/báo giả theo trạm-ngày, xác minh bộ lọc nguồn; bổ sung dữ liệu trong thí nghiệm riêng và cần tập xác nhận độc lập. Chưa triển khai cảnh báo thật.

**Đọc tổng hợp:** [Kết quả chẩn đoán và delta](analysis/KET_QUA_CHAN_DOAN_VA_DELTA.md). Lệnh chạy lại/kiểm tra: [TRAINING_README.md](analysis/TRAINING_README.md). Mỗi lần chẩn đoán/train tạo thư mục mới; không tự chạy lại chỉ vì đọc tài liệu.

## 14. Cập nhật: nhánh raw riêng trên ba điểm

**Ngày:** 10/11/2026. **Run:** `analysis/training_runs/20261110T043429_532355Z_raw_delta`.

### Thiết kế

- Giữ ba trạm core, mốc chia 01/01/2023 và 01/07/2023, embargo 24h, bước 5 phút, 12 feature, bốn cấu hình XGBoost delta và seed/trọng số. Không thêm bảy trạm, không đọc test hoặc cache nguồn từ 07/2023 trở đi.
- Đầu vào là trung vị raw trong bin đã kết thúc. Raw âm quy về 0 và có cờ/số lượng riêng; không tự loại giá trị cao hoặc bước nhảy mạnh.
- Bin trống hoặc raw ở hàng vật lý cuối null/không hữu hạn thì thiếu. Không fill/nội suy hoặc dùng mẫu tương lai để sửa lịch sử.
- **Nhãn vẫn là proc cuối bin tương lai (t+5,t+10]**, chỉ dùng cho giám sát hồi cứu. Proc không vào feature hoặc điều kiện lịch sử raw.
- Đã kiểm tra prefix replay: feature tại mốc train không đổi khi xử lý thêm dữ liệu tương lai. Điều này chỉ xét theo đồng hồ archive, chưa tái dựng thời điểm nhận bản tin/hiệu chuẩn nguồn.

### Cửa sổ và kết quả

- Train raw: **270.312**, ít hơn proc đúng một cửa sổ ở cuối nguồn `widely_proud_lizard` do bin nhãn tương lai chưa hoàn tất theo quy tắc nhánh raw. Nhãn cũ của mẫu đó <10 cm; không loại mẫu vì sai số model.
- Validation raw: **92.215**, trùng toàn bộ cửa sổ proc. Nhãn ≥10 cm giữ nguyên: **1.220 train, 1.037 validation**.
- Chọn `raw_delta_d3_weight20` theo MAE cân bằng trên toàn validation raw, trước khi tính bảng đối chiếu raw/proc.

| Chỉ số validation | Raw persistence | Raw delta | Proc delta — đối chứng cũ |
|---|---:|---:|---:|
| MAE cân bằng cm | 0,5309 | **0,4949** | 0,4725 |
| MAE nhãn ≥10cm | 0,9500 | **0,8960** | 0,8450 |
| MAE toàn tập cm | 0,1212 | **0,1029** | 0,1083 |
| Cửa sổ báo giả | **70** | 246 | 324 |
| Cửa sổ bỏ sót | 83 | **57** | 62 |

Raw delta giảm MAE cân bằng khoảng **6,8%** so với raw persistence nhưng báo giả tăng 70→246. Không thắng toàn diện và chưa chứng minh dự báo trực tiếp trước 10 phút hoặc chuyển sang trạm ESP32. Nhãn proc vẫn là tham chiếu hồi cứu chưa xác minh hiện trường.

### Kiểm tra và artifacts

- 14 kiểm thử raw mới; bộ test `analysis/` có 33 kiểm thử đạt (19 cũ + 14 mới). Cộng 16 test pipeline/đối soát 5 phút, tổng 49 kiểm thử không trùng nhau đều đạt.
- Dựng lại toàn bộ cửa sổ từ 51 cache có checksum; 33 fingerprint CSV, 6 file cửa sổ raw, 5 file cửa sổ proc khớp: PASS.
- Nạp lại bốn model raw và hai baseline, tái lập mọi dự đoán/chỉ số validation; năm phương pháp đối chiếu raw/proc và lựa chọn model khớp: PASS.
- Dấu vết cửa sổ và lý do loại: `window_coverage.csv`, `exclusion_reasons.csv`, `window_membership_local.parquet`; raw âm/mẫu cao còn giữ: `raw_bin_audit.csv`.
- Không chỉnh CSV/cache, kết quả proc, backend/frontend; chưa triển khai cảnh báo/định tuyến.

**Đọc tổng hợp:** [Kết quả nhánh raw](analysis/KET_QUA_NHANH_RAW.md). Hướng dẫn: [raw_branch/README.md](analysis/raw_branch/README.md). Mở rộng điểm nếu làm tiếp phải là thí nghiệm riêng, giữ nhánh raw này làm đối chứng và không âm thầm bỏ ca khó.

## 15. Cập nhật: mở rộng dữ liệu train nhánh raw

**Ngày:** 10/11/2026. **Run:** `analysis/training_runs/20261110T050356_109133Z_raw_expanded`.

### Chọn điểm và dữ liệu

- Kiểm tra đúng bảy ứng viên trước đó, chỉ đọc cache các tháng trước 01/2023 cho phần bổ sung; không đọc validation các điểm mới hoặc test.
- Chốt trước fit: ≥1.000 cửa sổ, ≥30 ngày, ≥20 nhãn cao, ≥1.000 nhãn thấp, ≥3 ngày và ≥3 nhóm nhãn cao cách nhau hơn sáu giờ. Nhóm thời gian không phải sự kiện độc lập đã xác minh.
- Cả bảy đạt độ phủ: `daily_gentle_beetle`, `evenly_divine_dingo`, `overly_heroic_squid`, `simply_pet_joey`, `tired_red_monkey`, `weekly_fair_dove`, `weekly_poetic_guinea`.
- Đạt độ phủ không xác nhận nhãn sạch; các điểm vẫn có cờ raw âm, mức cao kéo dài, dao động hoặc khác mốc. Lần này giữ TẤT CẢ cửa sổ raw hợp lệ để thử nghiệm, không dùng mask offline hoặc chỉ lấy ca đẹp. Không nâng bảy điểm thành core đã xác minh.
- Train core **270.312** không đổi; bổ sung **271.905**; tổng **542.217**. Nhãn train ≥10cm tăng **1.220→7.789**.
- Tổng là **10 mã triển khai**, không chắc 10 vị trí độc lập: `overly_heroic_squid` và `widely_proud_lizard` cùng tên 9th St/Smith St ở hai giai đoạn khác nhau.
- Giữ đúng **92.215** cửa sổ validation cũ, cùng input/nhãn, **1.037** nhãn cao tại hai trạm core. Không refit model ba mã; giữ preprocessing, nhãn proc tương lai, feature, bốn cấu hình, seed/trọng số và mốc chia tập.

### Kết quả trên cùng validation

| Chỉ số | Raw persistence | Raw delta 3 mã | Raw delta 10 mã |
|---|---:|---:|---:|
| MAE cân bằng cm | 0,5309 | **0,4949** | 0,5402 |
| MAE nhãn ≥10cm | 0,9500 | **0,8960** | 0,9539 |
| MAE toàn tập cm | 0,1212 | **0,1029** | 0,1357 |
| Cửa sổ báo giả | **70** | 246 | 133 |
| Cửa sổ bỏ sót | 83 | **57** | 70 |

- Ứng viên tốt nhất của cả hai nhóm đều là `d3_weight20`. Mở rộng tăng MAE cân bằng **0,04524cm, khoảng 9,1%**; giảm báo giả nhưng tăng bỏ sót. Cả bốn cặp cấu hình đều kém hơn về MAE cân bằng sau mở rộng.
- Phương pháp được chọn vẫn là **`three_d3_weight20`**, tương ứng model cũ `raw_delta_d3_weight20.ubj`. Không đổi tiêu chí sau khi xem kết quả hoặc tự thay model cũ.
- Chưa kết luận nguyên nhân do nhãn lỗi, khác trạm/hiệu chuẩn hay động lực nước. Kết quả không chứng minh mọi cách thêm điểm đều vô ích; validation cố định cũng chưa kiểm tra khả năng dự báo tại các điểm mới.

### Kiểm tra và artifacts

- Tám kiểm thử mở rộng mới; **57 kiểm thử đạt**: 41 trong `analysis/`, 16 pipeline/đối soát 5 phút.
- Dựng lại bảy bộ cửa sổ train/kiểm kê/lý do loại từ **62 cache có checksum**, xác minh **27 fingerprint CSV**, quyết định chọn trạm khớp: PASS.
- Nạp lại tám model delta và hai baseline, tái lập mọi dự đoán/chỉ số của 92.215 cửa sổ validation, bảng so từng cấu hình/từng trạm và lựa chọn model: PASS.
- Checksum input/model cũ không đổi. Không chỉnh CSV/cache, cửa sổ/model cũ, backend/frontend hoặc cảnh báo/định tuyến.

**Đọc tổng hợp:** [Kết quả mở rộng raw](analysis/KET_QUA_MO_RONG_RAW.md). Hướng dẫn: [raw_expansion/README.md](analysis/raw_expansion/README.md). Tiếp theo nên phân tích khác biệt ở dữ liệu bổ sung và đánh giá cảnh báo theo đợt; không tiếp tục thêm điểm/đổi tham số chỉ để tìm điểm validation đẹp. Vẫn cần kiểm định độc lập trước khẳng định vận hành thật.

## 16. Cập nhật: phân tích tác động mở rộng raw — phần A

**Ngày:** 11/11/2026. **Run:** `analysis/diagnostic_runs/20261111T000848_963785Z_raw_expansion`.

- Chỉ phân tích dữ liệu/model đã lưu; không fit, không đổi lựa chọn model, không đọc test hoặc validation các trạm mới. Các cache so sánh raw/proc chỉ thuộc các tháng trước 01/2023.
- Tỷ lệ nhãn cao trong train cũ 0,45%, nhóm bổ sung 2,42%; phần tổng trọng số ×20 của nhãn cao tăng 8,31%→22,57% sau gộp. P99 độ lớn delta tăng từ 1,0 cm ở core lên 18,4 cm ở nhóm bổ sung.
- Một số mã bổ sung có chênh raw/proc cùng bin lớn; đây là khác biệt phân bố, không tự chứng minh nhiễu/nhãn lỗi hay hiệu chuẩn sai.
- Khoảng 92,3% mức tăng MAE cân bằng tập trung ở `ugliest_cyan_elephant`. Hai pha đóng góp tăng lớn nhất là giữ mức cao và mức thấp ít biến đổi theo định nghĩa hồi cứu.
- Mở rộng sửa 135 báo giả nhưng tạo 22 báo giả mới; sửa một bỏ sót nhưng tạo 14 bỏ sót mới. Phát hiện cửa sổ chuyển ngưỡng giảm 26/83→16/83 dù MAE nhóm chuyển ngưỡng cải thiện.
- Tái lập tám model + hai baseline trên 92.215 cửa sổ validation: PASS. Phân rã cộng khớp tổng; 56 file trong hai run cũ giữ nguyên checksum. Chưa tách được nguyên nhân nhân quả; không xóa trạm/đoạn vì model sai.
- 63 kiểm thử đạt: 47 trong `analysis/` và 16 pipeline/đối soát 5 phút; có sáu kiểm thử mới. Validator phần A chỉ đọc kiểm tra 16 checksum đầu ra, 114 checksum đầu vào và các phân rã ghép cặp: PASS.
- Đã cập nhật đường dẫn tài liệu, thêm ánh xạ provenance trong bộ nhớ; loader cũ qua ánh xạ đọc đúng số hàng/checksum. Không sửa provenance/script model cũ, CSV/cache, backend/frontend.
- Bước tiếp theo là phần B: định nghĩa và đánh giá cảnh báo theo đợt; chưa thực hiện và chưa triển khai cảnh báo AI thật.

**Đọc tổng hợp:** [Phân tích mở rộng raw — phần A](analysis/KET_QUA_PHAN_TICH_MO_RONG_RAW.md).

## 17. Cập nhật: đánh giá cảnh báo theo đợt — phần B

**Ngày:** 11/11/2026. **Run:** `analysis/diagnostic_runs/20261111T004533_510057Z_warning_events`.

- Chốt quy tắc trước khi nạp validation: ngưỡng 10 cm, đoạn nhãn cao liên tiếp bước 5 phút, không nối qua nhãn thấp/khoảng thiếu; giữ cả đoạn một mẫu và cờ thiếu đầu/cuối.
- Có 67 đoạn vượt ngưỡng quan sát, không phải 67 trận ngập độc lập đã xác minh; 33 đoạn có ≥2 mẫu cao, 55 có onset quan sát và 53 đủ hai đầu. Độ phủ 320,19 ngày-trạm quan sát tương đương từ 92.215 cửa sổ.
- Bật từ một mẫu cao: persistence phát hiện 30/67, 23 đợt báo giả; xu hướng 40/67, 153 báo giả; raw ba mã 39/67, 145 báo giả; raw mười mã 35/67, 64 báo giả.
- Raw ba mã phát hiện thêm chín đoạn so với persistence, nhưng thêm 122 đợt báo giả. Chưa có ưu thế toàn diện cho cảnh báo.
- Độ nhạy chờ hai dự báo cao liên tiếp (bật từ mẫu thứ hai): raw ba mã giảm báo giả 145→33 nhưng giảm phát hiện 39→25. Không tự áp dụng chính sách này vào ứng dụng.
- Lead time là khoảng hồi cứu theo timestamp nhãn proc và quyết định khớp đầu tiên, không dùng thời điểm bắt đầu cảnh báo sai trước đó; chưa xác nhận báo trước trực tiếp 10 phút. Không tính các đoạn thiếu onset vào trung vị lead.
- Tái lập hai model + hai baseline; 24 hàng chỉ số khớp ledger và tổng theo trạm; FP/FN một mẫu khớp kết quả trước. 56 file trong hai run cũ không đổi checksum. Không nạp bảng train/test/cache, không refit, không sửa backend/frontend.
- 70 kiểm thử đạt: 54 trong `analysis/` và 16 pipeline/đối soát 5 phút, có bảy test sự kiện mới.
- Giữ lựa chọn model nghiên cứu theo MAE; chưa phê duyệt cảnh báo vận hành. Tiếp theo cần chốt mục tiêu/mức báo giả-bỏ sót chấp nhận được và tập xác nhận độc lập trước khi tối ưu/tích hợp tiếp.

**Đọc tổng hợp:** [Kết quả cảnh báo theo đợt](analysis/KET_QUA_CANH_BAO_THEO_DOT.md).

## 18. Cập nhật: bản đồ dự báo độ sâu và thí nghiệm objective MAE

**Ngày:** 11/11/2026. **Run thử nghiệm:** `analysis/training_runs/20261111T010522_872558Z_depth_mae_trial`.

- Thêm frontend `/forecast`: đọc file phát lại cục bộ, không gọi API live, không upload/lưu public. Dùng tọa độ trạm NYC từ metadata, không gán sang Việt Nam; không tham gia cảnh báo/định tuyến thật.
- Hiển thị current raw median, độ sâu dự báo, tăng/giảm, mức gần ngưỡng 8–<10 cm và baseline. Timeline bước 5 phút, phát nhanh ×400; mốc không có cửa sổ validation hợp lệ để trống, không fill. Eligibility hồi cứu có dùng nhãn để chấm điểm, không phải mô phỏng đầy đủ tính khả dụng online.
- Export minh họa model cũ: `analysis/replay_exports/20261111T010801_397964Z/forecast-replay-local.json`, thư mục được gitignore. Ba ngày minh họa, không xuất quan trắc/nhãn tương lai theo từng frame. Không tự phân phối dữ liệu phái sinh.
- Phân tích quanh ngưỡng: trên 839 nhãn 8–<12 cm, model cũ MAE 1,1803 cm so persistence 0,9899 cm; MAE toàn tập tốt hơn không đồng nghĩa quanh ngưỡng tốt hơn. Khi dự báo 8–<12 cm, P90 lỗi hồi cứu 3,7652 cm; không phải khoảng tin cậy từng lần.
- Một thử nghiệm cố định chỉ đổi objective `reg:squarederror→reg:absoluteerror`, giữ d3/250 cây/weight20, train 270.312, validation 92.215, feature/nhãn/preprocessing/mốc chia nguyên bản. Không quét tham số, đọc test hoặc refit bằng validation.
- MAE toàn tập giảm 0,10287→0,07475 cm (khoảng 27,3%); quanh nhãn 8–<12 giảm 1,18026→1,01381 cm. Nhưng MAE nhãn cao tăng 0,89604→0,92480 cm, sai số cực đại tăng 107,01→133,93 cm; MAE cân bằng 0,49494021→0,49494045 cm gần như hòa nhưng hơi kém.
- Giữ model cũ cho demo, model thử nghiệm lưu riêng; không tuyên bố cải thiện toàn diện hoặc đã xác nhận độc lập. Tái lập bốn phương pháp và nhóm lỗi, checksum 56 file hai run cũ: PASS.
- Build/test frontend, test export/MAE và kiểm tra Chrome với file thật; thử trạng thái gần ngưỡng, mốc trống, file sai schema xóa dự báo cũ, không gọi endpoint live. Có lỗi tải tile do mạng, đã thêm thông báo; bản đồ nền vẫn cần kết nối hoạt động.
- 80 kiểm thử đạt: 58 trong `analysis/`, 16 pipeline/đối soát 5 phút và sáu frontend; build TypeScript/Vite PASS.
- Chưa sửa backend/MQTT hoặc chạy inference trên ESP32. Cần tập xác nhận độc lập trước khi tối ưu/chọn model tiếp hoặc hiệu chuẩn khoảng dự báo.

**Đọc tổng hợp/cách mở:** [Bản đồ dự báo và độ chính xác](analysis/KET_QUA_BAN_DO_DU_BAO_DO_SAU.md).
