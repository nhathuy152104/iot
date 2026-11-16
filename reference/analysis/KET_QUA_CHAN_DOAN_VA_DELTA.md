# Kết quả chẩn đoán và thí nghiệm dự báo mức thay đổi

**Ngày:** 10/11/2026. **Trạng thái:** nghiên cứu hồi cứu, chưa triển khai API/frontend/ESP32.

## Kết luận ngắn

Đã tìm được nơi tập trung sai số của model cũ và train một thí nghiệm có kiểm soát. **Delta XGBoost vượt nhẹ persistence về MAE cân bằng trên validation, nhưng vẫn báo giả nhiều hơn rõ rệt. Chưa có bằng chứng đủ để đưa vào cảnh báo thực tế.**

## 1. Chẩn đoán model cũ

- Dự đoán lại model đã lưu, kiểm tra checksum nguồn và tái lập mọi chỉ số validation/test: **PASS**.
- Train có cực đại nhãn 97,8 cm, riêng trạm test chính là 62,8 cm; test có mức 118,6 cm.
- 27 cửa sổ có nhãn vượt cực đại train chiếm khoảng **72,8% tổng sai số tuyệt đối ở nhóm nhãn ≥10 cm của XGBoost cũ**.
- Trọng số ưu tiên nhãn ngập làm giảm bỏ sót nhưng tăng báo giả trong cả hai cặp cấu hình cũ.
- Trực tiếp xem 6 biểu đồ nguồn/dự báo: có đoạn tăng/rút khá trơn, nhưng cũng có mức cao kéo dài, chuyển mức đột ngột và raw/filt/proc khác biệt. Chưa có xác nhận hiện trường; không loại ca khó hoặc tự sửa nhãn.

Chi tiết:
- [Báo cáo chẩn đoán](diagnostic_runs/20261110T040149_598786Z/BAO_CAO_CHAN_DOAN.md).
- [Nhận xét trực quan và kết luận từng giả thuyết](diagnostic_runs/20261110T040149_598786Z/NHAN_XET_TRUC_QUAN.md).

## 2. Thí nghiệm chỉ đổi đích học

```text
Đích học: độ sâu tương lai − độ sâu hiện tại
Dự báo:  max(0, độ sâu hiện tại + mức thay đổi do model dự báo)
```

Giữ nguyên 270.313 cửa sổ train, 92.215 validation, 3 trạm, 12 feature, 4 cấu hình, seed, trọng số mẫu và quy tắc chọn. Không dùng mask offline, không thêm bảy trạm, không refit trên validation. Trọng số vẫn dựa vào độ sâu tương lai ≥10 cm, không dựa vào delta.

**Không nạp hoặc dự đoán test trong thí nghiệm delta.** Test cũ chỉ được đọc để chẩn đoán model cũ ở bước trước.

### Chỉ số trên cùng validation

| Chỉ số | Persistence | XGBoost cũ | Delta XGBoost được chọn |
|---|---:|---:|---:|
| MAE cân bằng (cm) | 0,4851 | 0,5130 | **0,4725** |
| MAE khi nhãn ≥10cm (cm) | 0,9104 | 0,8994 | **0,8450** |
| MAE toàn tập (cm) | **0,0693** | 0,1352 | 0,1083 |
| Cửa sổ báo giả | **78** | 336 | 324 |
| Cửa sổ bỏ sót | 79 | **59** | 62 |

Chọn `delta_d3_weight20` theo tiêu chí MAE cân bằng đã định. Cải thiện so với persistence là **0,01260 cm, khoảng 2,6%** trên chỉ số đó. Đây là cải thiện nhỏ, không đồng nghĩa model tốt hơn trên mọi tiêu chí; báo giả vẫn khoảng 4,15 lần persistence. Không đổi tiêu chí sau khi xem kết quả để chọn cấu hình khác.

Chi tiết: [BAO_CAO_DELTA.md](training_runs/20261110T040220_424848Z_delta/BAO_CAO_DELTA.md).

## 3. Kiểm tra kỹ thuật

- 12 kiểm thử tổng hợp mới: xử lý delta âm, clipping sau khi cộng độ sâu, lựa chọn cho phép baseline thắng, chặn test, phân tầng, phân đoạn và bảo vệ checksum.
- Bộ test gốc trong `analysis/` cùng test mới: **19 kiểm thử đạt**.
- Bộ test pipeline 5 phút và đối soát 18 trạm: **16 kiểm thử đạt**; tổng hai nhóm là **35 kiểm thử** không trùng nhau.
- Nạp lại cả 4 model delta, model tuyệt đối cũ và hai baseline; tính lại toàn bộ 92.215 dự đoán validation cho mỗi phương pháp; JSON/CSV và quyết định chọn model khớp: **PASS**.
- 5 file đầu vào train/validation có checksum và số hàng trùng lượt gốc.
- Phân rã chẩn đoán khớp tổng theo pha và theo phạm vi train; 7 file nguồn, 3 file cache và 6 biểu đồ được kiểm tra: PASS (`diagnostic_validation.json`).
- CSV gốc, cửa sổ dữ liệu, model/artifacts lượt đầu và backend/frontend không bị chỉnh sửa.

Kiểm tra kỹ thuật không chứng minh nhãn đúng, bộ lọc causal hoặc an toàn vận hành.

## 4. Tiếp theo nên làm gì?

1. **Chưa triển khai cảnh báo thật.** Giữ cảnh báo theo ngưỡng đo hiện tại độc lập với AI.
2. Định nghĩa trước tiêu chí cảnh báo theo sự kiện: số lần báo giả/trạm-ngày, sự kiện bỏ sót và lead time thực. Đếm cửa sổ hiện tại chưa trả lời các câu hỏi đó.
3. Xác minh bộ lọc `depth_proc_mm` có sử dụng tương lai và trễ bao lâu. Chưa rõ thì tiếp tục gọi đây là đánh giá hồi cứu.
4. Thí nghiệm bổ sung dữ liệu từ nhóm trạm đã đối soát phải làm riêng; không thay đồng thời mục tiêu/feature/trọng số để giữ khả năng giải thích kết quả.
5. Cần bộ kiểm định độc lập chưa dùng lựa chọn. Validation và test cũ đã được khảo sát nên không dùng để khẳng định thành công cuối cùng.
6. Nếu cần demo sớm, có thể xây chế độ phát lại có nhãn “thử nghiệm”, hiển thị cả baseline và AI; không ghép NYC với ESP32 thành một chuỗi giả.

Nguồn: FloodNet (New York University and The City University of New York). Artifacts giữ cục bộ; làm rõ quyền trước phân phối dữ liệu/trọng số. Phân tích/biểu diễn phái sinh được phép theo CC BY-NC-SA 4.0 cùng Data Access License Agreement.
