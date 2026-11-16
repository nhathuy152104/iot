# Mở rộng sàng lọc pluvial — bước 5 phút

Đọc toàn bộ CSV nguồn nhưng chỉ trích các deployment được audit phân loại `pluvial`. Không sửa nguồn và không train XGBoost. Dữ liệu đầu ra chỉ là bộ thăm dò, chưa xác minh bộ lọc proc có causal hay không.

## Chạy

```powershell
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\five_minute\pipeline.py"
```

Dùng môi trường Python/thư viện của `analysis/requirements.txt`. Chạy lại sẽ dùng cache theo fingerprint nguồn (size + mtime + danh sách trạm + schema). Giữ nguyên nguồn khi chạy; fingerprint không thay cho checksum toàn nội dung. Không tự xóa cache cũ hoặc artifact của lượt chạy trước. Mỗi lượt xuất `selection_manifest.json` xác định các đầu ra hiện hành.

## Kiểm thử

```powershell
cd "D:\CODE\IOT\IOT training\analysis\five_minute"
& "..\.venv\Scripts\python.exe" -m unittest test_pipeline.py -v
```

## Phương pháp

- Lưới UTC 5 phút `(t-5,t]`. Lấy nguyên hàng cuối (kể cả null), không lấy last-non-null độc lập cho ba cột. Không fill/nội suy.
- Lịch sử 3 bin t-10,t-5,t, nhãn bin t+10. Support đầu vào rộng 15 phút. Lưu độ mới số đo để không giấu nhịp gửi khác nhau.
- Chỉ nhận cửa sổ đủ 3 bin proc không âm và nhãn không âm. Thiếu ở khoảng tương lai giữa hiện tại và nhãn không tự loại, vì không dự báo cực đại cả khoảng.
- Đặc trưng giới hạn theo whitelist `feature_columns` trong manifest. **KHÔNG dùng `audit_*`, `review_*`, nhãn hay metadata làm đầu vào tự động.** Đặc biệt `audit_*` có thể chứa thời điểm thực của nhãn tương lai.
- Train trước 01/01/2023, validation trước 01/07/2023, test sau đó. Purge toàn support và embargo 24 giờ mỗi bên ranh giới. Embargo là quy ước thận trọng, không chứng minh độc lập cơn mưa và có thể làm mất sự kiện hiếm.
- Cấu hình được đặt trước chạy; chỉ dùng train để phân nhóm/xếp hạng trạm, không ép đủ 10–20. Mọi cờ được giữ lại trong cửa sổ; sàng lọc cấp trạm không phải xác minh ngập hiện trường.
- Trạm muộn/ít sự kiện/chất lượng cần xem vẫn được xuất cửa sổ và giữ trong danh sách, không xóa.
- Test 2023 đã có khảo sát mô tả ở lượt trước; không gọi đây là holdout xác nhận chưa từng xem. Cần dữ liệu độc lập cho khẳng định cuối cùng, đánh giá thêm theo mùa/sự kiện.

Mở `BAO_CAO_5_PHUT.md` và `index.html` sau chạy. Biểu đồ được chọn từ train và chỉ dùng quan trắc thật, không nối qua khoảng thiếu.

`manual_review_notes.json` ghi nhận đối soát trực quan bổ sung của các ca train đã xem; không thay đổi tự động ngưỡng và không chứng nhận nhãn sạch. `validate_outputs.py` kiểm tra toàn bộ file Parquet (độ hữu hạn đặc trưng, support chia tập, độ mới số đo và lead time thật, số hàng và phân nhóm train). Kết quả ở `output_validation.json` chỉ là kiểm tra cấu trúc, không xác minh ngập thực.

## Quyền sử dụng

Nguồn: FloodNet (New York University and The City University of New York).

Giữ tất cả cache, Parquet cửa sổ và các đầu ra nhạy cảm cục bộ. Không upload/chia sẻ dữ liệu gốc hoặc trích xuất khi chưa có quyền. Phân tích/biểu diễn phái sinh được phép phải tuân CC BY-NC-SA 4.0 cùng Data Access License Agreement. Không suy quyền phân phối trọng số model từ quyền chia sẻ code. Trợ lý phụ chỉ được yêu cầu góp ý phương pháp, không đọc hoặc nhận quan trắc nguồn.
