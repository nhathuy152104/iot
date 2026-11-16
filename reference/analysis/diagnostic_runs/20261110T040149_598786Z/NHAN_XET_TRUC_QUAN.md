# Ghi chú sau khi trực tiếp xem 6 biểu đồ chẩn đoán

**10/11/2026. Chỉ đối soát hình thái, không xác nhận ngập thật hoặc sửa nhãn.**

## Kết quả có thể kiểm tra bằng số

- Model cũ và baseline đã được dự đoán lại; toàn bộ chỉ số validation/test khớp lượt gốc.
- Cực đại nhãn train toàn core: **97,8 cm**; riêng `ugliest_cyan_elephant`: **62,8 cm**. Test có mức **118,6 cm**.
- **27** cửa sổ test có nhãn >97,8 cm. Nhóm này đóng góp **1.478,95 / 2.032,77 ≈72,8%** tổng sai số tuyệt đối của XGBoost cũ trên các nhãn ≥10 cm.
- Trong nhóm còn lại (nhãn không vượt cực đại train), MAE nhãn ≥10 cm của XGBoost cũ là **5,035 cm**, so với **5,435 cm** của persistence. Đây chỉ là phân rã hồi cứu, KHÔNG loại nhóm cực trị khỏi điểm chính hoặc dùng làm điều kiện dự báo.
- Validation weight20 so với uniform cùng d3: FP tăng **47→336**, FN giảm **98→59**; cùng d4: FP tăng **58→525**, FN giảm **98→54**. Có đánh đổi báo giả/bỏ sót rõ rệt.

## Quan sát trực tiếp

| Ca UTC, cùng trạm `ugliest_cyan_elephant` | Quan sát | Điều chưa được kết luận |
|---|---|---|
| Validation 01/06–02/06/2023 | Mức khoảng 12–16 cm giữ hơn một ngày, cuối đoạn giảm rất đột ngột; raw/filt/proc phần lớn trùng. | Chưa xác định nước thật, vật cản hoặc thay đổi mốc. Trùng raw/proc không phải xác nhận nhãn sạch. |
| Validation 27/06/2023 | Dao động mạnh ở mức vài cm đến khoảng 16 cm; các mô hình bám dao động nhưng có báo cao quá. | Chưa phân biệt được nhiễu phản xạ với biến động nước thật. |
| Validation 29/04/2023 | Tăng dần tới gần 30 cm rồi rút dần, có đỉnh phụ; các mô hình khá gần nhau. | Hình thái phù hợp để nghiên cứu nhưng chưa phải chứng cứ hiện trường. |
| Test 23/08/2023 | Raw/filt có mức cao gần 118 cm; proc chuyển từ gần 0 sang mức cao rồi về gần 0. Ở đoạn giữ cao, XGBoost cũ thường chỉ khoảng 40–80 cm trong khi persistence bám mức đang đo. | Có bằng chứng model hụt ở mức cao ngoài train, đồng thời hình thái proc cần đối soát. Không tự kết luận là ngập 1,18 m hoặc tự xóa ca này. |
| Test 18/08/2023, đoạn 09:50 | Raw có nhiều giá trị âm rất lớn; filt có mức dương trong khi proc đa phần 0 và chỉ giữ một đoạn ngắn. | Cần hiểu xử lý nguồn/hiệu chuẩn; không tự sửa dữ liệu hoặc coi raw là chân lý. |
| Test 18/08/2023, đoạn 08:00 | Có tăng mức rồi raw trở nên âm lớn; filt tiếp tục dương trong khi proc về 0. Dự báo quanh điểm chuyển dễ sai. | Chưa thể xác nhận diễn biến nước hay nguồn lỗi bằng ảnh. |

## Kết luận theo giả thuyết

1. **Thiếu độ phủ mức cao/ngoại suy độ sâu tuyệt đối:** được hỗ trợ mạnh bởi phân rã sai số và biểu đồ test. Chưa chứng minh là nguyên nhân duy nhất.
2. **Trọng số mẫu ngập làm tăng báo giả:** được hỗ trợ bởi cả hai cặp cấu hình trên validation cũ.
3. **Ít đoạn/trạm chi phối:** được hỗ trợ; 131/137 nhãn test ≥10 cm ở cùng một trạm. Những cửa sổ tương quan không phải các sự kiện độc lập.
4. **Nhãn nhiễu hoặc bộ lọc nguồn nhìn tương lai:** còn nghi vấn, CHƯA xác minh. Không có thời tiết, hiện trường hoặc code bộ lọc để kết luận.

Đã giữ nguyên toàn bộ cửa sổ. Thí nghiệm delta được chốt theo đề xuất trước đó, giữ nguyên dữ liệu/feature/cấu hình và chỉ chọn bằng validation; không dùng các ca test này để chọn ngưỡng hay loại dữ liệu.

Nguồn: FloodNet (New York University and The City University of New York). Giữ cục bộ. Phân tích/biểu diễn phái sinh được phép theo CC BY-NC-SA 4.0 và các điều khoản Data Access License Agreement; không phân phối dữ liệu gốc hoặc artifact chưa làm rõ quyền.
