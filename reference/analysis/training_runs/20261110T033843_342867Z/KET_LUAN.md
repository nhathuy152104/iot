# Kết luận lượt train đầu

**Đã cài và train thành công về kỹ thuật. XGBoost chưa tốt hơn persistence theo tiêu chí sai số đã đặt, kể cả trên validation. Không đưa model này vào cảnh báo thực tế.**

## Thí nghiệm

- Python 3.15.2; XGBoost 3.4.1; scikit-learn 1.9.1. CPU, tối đa 4 luồng.
- 270.313 cửa sổ train từ 3 trạm; 92.215 validation; 42.218 test. Validation/test thực tế chỉ có 2 trạm còn dữ liệu; widely_proud_lizard chỉ đóng góp train.
- 4 cấu hình XGBoost, chốt trước fit. Chọn `d3_weight20`: depth3, 250 cây, learning rate0,05; trọng số20 cho nhãn train>=10cm.
- Model được chọn là tốt nhất **trong bốn cấu hình XGBoost**, không phải thắng baseline. Không refit hoặc thay cấu hình theo test.
- Dùng toàn bộ cửa sổ gốc của nhóm core, không lọc theo mask đối soát offline.

## Sai số và vượt ngưỡng

| Chỉ số test | Persistence | Xu hướng tuyến tính | XGBoost |
|---|---:|---:|---:|
| MAE toàn tập (cm) | 0,0913 | 0,1436 | 0,1848 |
| MAE khi nhãn >=10cm (cm) | 5,51 | 9,07 | 14,84 |
| MAE cân bằng hai nhóm (cm) | 2,79 | 4,59 | 7,49 |
| Recall nhãn >=10cm | 68,6% | 70,8% | 75,9% |
| Precision nhãn >=10cm | 68,6% | 42,4% | 40,3% |
| Số cửa sổ báo giả | 43 | 132 | 154 |
| Số cửa sổ bỏ sót | 43 | 40 | 33 |
| Phát hiện cửa sổ hiện tại <10, tương lai >=10cm | 0/43 | 9/43 | 10/43 |

Validation MAE cân bằng: persistence **0,4851cm**, XGBoost được chọn **0,5130cm**. Không có bằng chứng thắng baseline ngay từ validation; kết quả test không đảo ngược kết luận đó.

Recall cao hơn không đồng nghĩa mô hình tổng thể tốt hơn: XGBoost đánh đổi thêm báo giả và sai số lớn hơn. Chỉ 137 nhãn test >=10cm; 131 thuộc ugliest_cyan_elephant. Các hàng liên tiếp tương quan, không phải 137 trận ngập độc lập. 43 cửa sổ chuyển ngưỡng cũng không phải 43 lần khởi phát độc lập và không cho biết lead time cảnh báo theo sự kiện.

## Quan sát sau train, không dùng để sửa model

Đã xem hai biểu đồ test sau chốt lựa chọn. Ở ca ugliest_cyan_elephant ngày 23/08/2023, nhãn proc tăng tới trên100cm và giữ mức cao; XGBoost thường dự đoán thấp hơn đáng kể trong đoạn này. Cần đối soát bản thân sự kiện/nhãn và độ phủ train trước khi kết luận nguyên nhân là thiếu khả năng ngoại suy hoặc thay đổi phân phối. Chỉ nhìn biểu đồ không xác nhận đây là ngập thực hay vật cản. Không xóa ca khó để nâng điểm test.

## Bước cải thiện hợp lý, chưa thực hiện

1. Làm rõ bộ lọc proc causal/độ trễ và các trường hợp mức cao nghi vật cản; không coi nhãn archive là ground truth đã kiểm định.
2. Xem thêm nhiều sự kiện train đa dạng; khi bổ sung dữ liệu, giữ nhóm nghi vấn riêng và báo cáo độ nhạy, không chỉ huấn luyện trên các ca đẹp.
3. Với nhiệm vụ cảnh báo sớm, xác định chỉ số theo sự kiện/khởi phát và mức bỏ sót/báo giả chấp nhận được, thay vì chỉ MAE hoặc recall theo cửa sổ.
4. Chỉ tối ưu tiếp bằng train/validation. Test này đã khảo sát; cần dữ liệu độc lập hoặc thiết kế backtest đã chốt trước để xác nhận các cải tiến sau này.

## Kiểm tra kỹ thuật

- 4 kiểm thử tổng hợp cho baseline, chỉ số và chọn cấu hình: PASS.
- `artifact_validation.json`: PASS; kiểm tra checksum7 file đầu vào không rỗng, nạp model, tính lại dự đoán42.218 hàng test, baseline và mọi chỉ số.
- Giữ nguyên CSV nguồn, pipeline/nhóm core, API/frontend; chưa có model được phê duyệt deployment.

Đọc `BAO_CAO_TRAIN.md`, `metrics.csv`, `test_breakdown.csv` và cấu hình trước fit để xem đầy đủ. Đây là nghiên cứu hồi cứu trên dữ liệu dẫn xuất cục bộ từ FloodNet (New York University and The City University of New York). Rà quyền trước chia sẻ dữ liệu dẫn xuất/trọng số. CC BY-NC-SA 4.0 và Data Access License Agreement áp dụng theo phạm vi tác phẩm phái sinh được phép.
