# Kết quả phân tích mở rộng raw — phần A

**Ngày:** 11/11/2026. **Không train lại, không đọc test, không đổi model được chọn.**

Run: [20261111T000848_963785Z_raw_expansion](diagnostic_runs/20261111T000848_963785Z_raw_expansion/BAO_CAO_PHAN_TICH_A.md).

## 1. Kết luận chính

Mở rộng dữ liệu làm thay đổi mạnh phân bố đầu vào/nhãn và đóng góp vào hàm mất mát. Đã định vị phần suy giảm, nhưng **chưa chứng minh nguyên nhân nhân quả là nhiễu, khác trạm hay hiệu chuẩn**. Không loại trạm hoặc đoạn chỉ vì sai số lớn. Giữ model raw ba mã theo tiêu chí cũ.

### Dữ liệu mới không chỉ là nhiều bản ghi hơn

| Chỉ số train | Ba mã cũ | Bảy mã bổ sung |
|---|---:|---:|
| Tỷ lệ nhãn ≥10 cm | 0,45% | 2,42% |
| P99 độ sâu raw hiện tại | 5,1 cm | 72,0 cm |
| P99 độ lớn delta (proc tương lai − raw hiện tại) | 1,0 cm | 18,4 cm |

Với trọng số ×20, nhóm nhãn cao chiếm **8,31%** tổng trọng số train cũ và **22,57%** tổng trọng số train mở rộng. Đây là tổng trọng số mẫu, không phải đo trực tiếp tỷ lệ gradient/loss. Cả hai cấu hình uniform cũng kém hơn sau mở rộng, nên không thể quy toàn bộ suy giảm cho ×20.

Chênh raw/proc cùng bin đáng chú ý tại `daily_gentle_beetle`, `simply_pet_joey`, `tired_red_monkey`, `weekly_fair_dove`: P99 độ lớn chênh lần lượt khoảng **43,1; 72,0; 30,0; 11,0 cm**. Raw ở đây là trung vị bin, proc là hàng cuối: chênh lệch bao gồm cả cách tổng hợp lẫn xử lý nguồn, không tự xác nhận cảm biến/nhãn lỗi.

## 2. Sai số tăng ở đâu?

MAE cân bằng tăng **0,045239 cm**:

- `ugliest_cyan_elephant`: đóng góp **+0,041751 cm**, khoảng **92,3%** mức tăng ròng.
- `daily_new_falcon`: đóng góp **+0,003488 cm**. Sai số nhóm nhãn cao tại trạm này giảm, nhưng sai số nhóm thấp tăng; chỉ có tám nhãn cao nên không khái quát từ riêng nhóm này.
- Theo pha, đóng góp lớn nhất là `stable_high` **+0,026283 cm** và `stable_low` **+0,017950 cm**. Pha được phân theo chênh proc tương lai với raw hiện tại; không phải giai đoạn thủy văn đã xác minh.
- Nhóm chuyển từ dưới 10 cm lên ≥10 cm giảm đóng góp sai số **−0,006551 cm**, nhưng số cửa sổ được phát hiện lại giảm **26/83 → 16/83**. Sai số độ sâu và phát hiện vượt ngưỡng không phải cùng một mục tiêu.

Các đóng góp dùng mẫu số toàn validation theo hai nhóm nhãn; không lấy trung bình MAE cân bằng của các trạm. Chi tiết theo trạm/pha/mức nước/tháng/cờ nằm trong `paired_error_decomposition.csv`.

## 3. Đổi chác báo giả và bỏ sót

Ghép đúng cùng cửa sổ của hai model `d3_weight20`:

- Sửa **135** báo giả cũ, tạo **22** báo giả mới → tổng **246 → 133**.
- Sửa **1** bỏ sót cũ, tạo **14** bỏ sót mới → tổng **57 → 70**.

Chưa phải số đợt cảnh báo hoặc mức báo trước; phần B chưa thực hiện.

## 4. Đối soát trực quan

Đã trực tiếp xem ba ngày đứng đầu đóng góp tăng sai số tại `ugliest_cyan_elephant`:

- **02/06/2023:** phần lớn ngày giữ khoảng 13–16 cm; dự báo mở rộng thường thấp hơn model ba mã. Có chuyển mức xuống thấp đột ngột ở cuối đoạn, chưa kết luận do nước hay xử lý/quan trắc.
- **23/04/2023:** có diễn biến tăng đến khoảng 28 cm rồi giảm; khác biệt hai model nhỏ ở nhiều điểm nhưng tích lũy thành chênh sai số theo ngày.
- **04/03/2023:** tăng đến khoảng 33 cm rồi giảm; không có bằng chứng từ biểu đồ này rằng chỉ cần thêm dữ liệu có mức cực đại cao hơn sẽ cải thiện dự báo.

Các biểu đồ vẽ nhãn tương lai theo **đồng hồ quyết định** để đối chiếu dự báo; không phải thời điểm thực của phép đo nhãn. Chỉ vẽ các cửa sổ hợp lệ, dùng điểm rời để không nối qua khoảng thiếu. Đây không phải xác minh hiện trường.

## 5. Kiểm tra và khả năng chạy lại

- Tái lập toàn bộ dự đoán/chỉ số của tám model và hai baseline trên **92.215** cửa sổ validation: PASS.
- Checksum cửa sổ/cache đã dùng khớp; các phân rã cộng lại đúng chênh MAE cân bằng.
- **56 file** trong hai run raw cũ giữ nguyên checksum trước/sau phân tích.
- Không refit, không đọc test hoặc validation của trạm mới. Cache phục vụ đối chiếu raw/proc chỉ đọc các tháng trước 01/2023 và cắt theo embargo train.
- Có một lượt chẩn đoán không hoàn tất trước đó, được đánh dấu `FAILED.md`; không dùng làm kết quả.
- **63 kiểm thử đạt:** 47 trong `analysis/` (gồm sáu kiểm thử mới) và 16 trong `analysis/five_minute/`. Lệnh validator chỉ đọc kiểm tra lại 16 checksum đầu ra, 114 checksum đầu vào và mọi phân rã ghép cặp: PASS.

```powershell
# Tạo run chẩn đoán A mới, không train và không ghi đè run cũ.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\diagnose_raw_expansion.py"

# Kiểm tra run hoàn chỉnh, không ghi file hoặc fit.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\validate_raw_expansion_diagnosis.py" "D:\CODE\IOT\IOT training\analysis\diagnostic_runs\20261111T000848_963785Z_raw_expansion"
```

## 6. Đường dẫn sau khi di chuyển thư mục

Các lệnh trong README đã chuyển sang `D:\CODE\IOT\IOT training`. JSON provenance cũ vẫn lưu đường dẫn lịch sử để giữ nguyên checksum.

- `archive_paths.py`: ánh xạ đường dẫn khi đọc cho chẩn đoán mới, kiểm tra tồn tại và checksum trước dùng.
- `run_relocated.py`: chạy entry point cũ với các đường dẫn JSON được ánh xạ **trong bộ nhớ**, không sửa script đã đóng băng hoặc bỏ kiểm tra checksum. README đã thêm wrapper vào các lệnh cần đọc provenance cũ.
- Wrapper giữ hành vi gốc của script: lệnh train vẫn train, validator có thể ghi lại báo cáo/validation artifact như trước. **Không phải chế độ chỉ đọc**; chưa chạy lại các validator có ghi file trên run cũ trong công việc này.
- Đã kiểm tra loader cũ qua lớp ánh xạ: đọc đúng **270.312 core train, 542.217 expanded train, 92.215 validation**, các checksum/code hash vẫn khớp.

**Tiếp theo:** chốt quy tắc và đánh giá theo đợt (phần B), thay vì tiếp tục tăng trạm hoặc chỉnh tham số để tìm điểm validation đẹp. Vẫn cần tập xác nhận độc lập trước khẳng định vận hành.

Nguồn: FloodNet (NYU/CUNY). Giữ dữ liệu/artifacts cục bộ; tuân giấy phép nguồn, làm rõ quyền trước chia sẻ.
