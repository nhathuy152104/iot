# Bản đồ dự báo độ sâu và thí nghiệm độ chính xác

**Ngày:** 11/11/2026. Hướng sản phẩm: độ sâu là thông tin chính; màu “gần ngưỡng” bổ trợ, không chỉ bật/tắt cảnh báo. Chưa chạy model trên ESP32 hoặc thay cảnh báo/định tuyến thật.

## 1. Bản đồ thử nghiệm đã có

Mở **http://localhost:5173/forecast** khi Vite đang chạy. Có liên kết từ bản đồ chính.

- Đọc file phát lại **cục bộ** do script export tạo; không cần backend, không upload file lên server, không lưu dữ liệu vào public/localStorage.
- Hiển thị đúng vị trí trạm NYC lấy theo `deployment_id` từ các cột metadata CSV tháng 01/2023; không gán trạm NYC sang tọa độ Việt Nam. Không gọi geocoder.
- Độ sâu raw hiện tại, độ sâu dự kiến, tăng/giảm dự kiến, baseline persistence tùy bật/tắt.
- Màu dưới 8 cm / gần ngưỡng 8–<10 cm / đạt-vượt 10 cm. Đây là **quy ước nghiên cứu**, không là phân loại đường an toàn. Không hạ trạng thái số đo hiện tại theo dự báo giảm.
- Thanh thời gian, ba kịch bản minh họa đã từng xem, phát nhanh ×400, dừng/về đầu/+5 phút.
- Mục tiêu vẫn là số đo cuối bin `(t+5,t+10 phút]`, không phải cực đại cả 10 phút hoặc giá trị tức thời đúng t+10.
- Mốc không có cửa sổ validation hợp lệ được để trống, không fill/giữ dự báo cũ. Eligibility validation có xét nhãn tương lai để chấm điểm, nên mốc trống **không nhất thiết là cảm biến mất số đo**; đây không phải mô phỏng đầy đủ tính khả dụng lúc vận hành live.
- Nêu rõ chưa hiệu chuẩn độ tin cậy, không dựng một khoảng tin cậy giả từ MAE. Thông tin P90 sai số hồi cứu khi dự báo 8–12 cm chỉ để tham khảo, không bảo đảm từng dự báo.

File đã tạo:

`replay_exports/20261111T010801_397964Z/forecast-replay-local.json`

File chứa ba ngày minh họa (mỗi ngày 288 mốc), chỉ current raw median, dự báo, baseline và metadata tổng hợp. **Không xuất proc/quan trắc tương lai vào từng frame.** Kịch bản không được dùng thay tập đánh giá đại diện. Export giữ nguyên model raw ba mã đã chọn, không dùng model thử nghiệm mới.

```powershell
# Tạo export cục bộ mới; không train hoặc đọc test.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\export_forecast_replay.py"
```

Thư mục export đã được gitignore; không đưa file vào `frontend/public`, không commit/chia sẻ trước khi làm rõ quyền nguồn. Việc đọc file dùng input trình duyệt, không phải endpoint upload. Bản đồ nền vẫn có request tile theo viewport, không gửi file quan trắc.

## 2. Kiểm tra độ chính xác quanh ngưỡng và ca sai lớn

Trên **cùng 92.215 cửa sổ validation**, model ba mã cũ:

- MAE cân bằng **0,49494 cm**, tốt hơn persistence **0,53088 cm**.
- Nhưng riêng **839 cửa sổ có nhãn 8–<12 cm**, MAE **1,1803 cm**, kém persistence **0,9899 cm**. Đây là nhóm theo nhãn để đánh giá, không là bộ lọc online.
- Khi **dự báo** nằm 8–<12 cm, có 1.124 cửa sổ, P90 độ lớn sai số **3,7652 cm**. Không nhầm nhóm này với 839 cửa sổ theo nhãn.
- Sai số tuyệt đối lớn nhất của model cũ là **107,0066 cm**; MAE nhỏ không loại trừ các lần sai lớn. Bảng 20 ca sai lớn được giữ cục bộ trong run thử nghiệm.

Vì vậy giao diện không mô tả dự báo 9 cm là “chắc chắn dưới 10 cm” hoặc đường an toàn.

## 3. Một thí nghiệm cải thiện có kiểm soát

Run: [20261111T010522_872558Z_depth_mae_trial](training_runs/20261111T010522_872558Z_depth_mae_trial/BAO_CAO_THI_NGHIEM_DO_SAU.md).

Giả thuyết: hàm học sai số tuyệt đối có thể phù hợp MAE và bớt bị ca cực đoan chi phối hơn sai số bình phương. **Chỉ đổi objective** `reg:squarederror → reg:absoluteerror`, giữ d3/250 cây/weight20, 12 feature, nhãn, preprocessing, train raw 270.312, validation 92.215, seed và mốc chia tập. Không quét tham số hoặc chọn trạm lại.

| Chỉ số validation | Persistence | Raw ba mã cũ | Thử nghiệm absolute error |
|---|---:|---:|---:|
| MAE toàn tập cm | 0,12123 | 0,10287 | **0,07475** |
| MAE cân bằng cm | 0,53088 | **0,49494021** | 0,49494045 |
| MAE nhãn ≥10cm | 0,94995 | **0,89604** | 0,92480 |
| MAE nhãn 8–<12cm | **0,98993** | 1,18026 | 1,01381 |
| Sai số tuyệt đối lớn nhất cm | 131,90 | **107,01** | 133,93 |

**Kết quả:** MAE toàn tập giảm khoảng **27,3%** so với model cũ và MAE quanh ngưỡng giảm khoảng **14,1%**, nhưng MAE nhãn cao tăng khoảng **3,2%**, sai số cực đại xấu hơn và MAE cân bằng gần như hòa nhưng hơi kém. Không kết luận cải thiện toàn diện, không đổi tiêu chí để chọn model mới.

**Giữ model cũ cho bản đồ thử nghiệm.** Model mới lưu riêng, chưa được duyệt. Chỉ đánh giá validation đã dùng nhiều lần; chưa đọc test hoặc refit bằng validation. Cần dữ liệu xác nhận độc lập trước kết luận tổng quát.

```powershell
# Một lượt thử nghiệm mới, không ghi đè run cũ.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\depth_accuracy.py"

# Validator chỉ đọc, tái lập dự đoán/chỉ số, không fit.
& "D:\CODE\IOT\IOT training\analysis\.venv\Scripts\python.exe" "D:\CODE\IOT\IOT training\analysis\validate_depth_accuracy.py" "D:\CODE\IOT\IOT training\analysis\training_runs\20261111T010522_872558Z_depth_mae_trial"
```

## 4. Kiểm tra

- **80 kiểm thử đạt:** 58 trong `analysis/`, 16 pipeline/đối soát 5 phút và sáu frontend; `npm run build` PASS. Không gồm lại các test trùng giữa hai lần chạy.
- Build TypeScript/Vite và test frontend bằng Node built-in test runner, không thêm dependency.
- Test xử lý ngưỡng 9/10 cm, thiếu dữ liệu, file sai schema/live, tọa độ/time không hợp lệ, baseline không khớp, làm tròn sát ngưỡng.
- Test export giữ đủ lưới 5 phút với mốc trống, không leak nhãn tương lai vào frame; test công thức MAE/bias và nhóm rỗng.
- Validator tái lập bốn phương pháp dự báo, toàn bộ chỉ số và nhóm lỗi; checksum output và 56 file trong hai run cũ giữ nguyên.
- Đã mở trang bằng Chrome, chọn file export thật, kiểm tra trạng thái gần ngưỡng và thiếu dữ liệu. File sai schema xóa trạng thái cũ thay vì giữ dự báo cũ. Không có request endpoint `/api/` từ trang forecast. Kiểm tra layout nhỏ không tràn ngang.
- Khi kiểm tra có lỗi DNS ở một số tile OSM (và favicon cũ 404), không phải lỗi inference/React. Trang đã thêm thông báo tải thiếu bản đồ nền; dữ liệu dự báo vẫn đọc cục bộ. Cần mạng/nhà cung cấp tile hoạt động để xem đủ nền đường phố.
- Có một lượt thử nghiệm dừng trước fit do ghi tham số NaN mặc định; đánh dấu `FAILED.md`, không dùng làm kết quả.

## 5. Hướng cải thiện tiếp

1. Giữ MAE cân bằng làm tiêu chí chính; xem kèm sai số nhóm ≥10 cm, quanh ngưỡng và ca sai lớn, không chỉ MAE toàn ngày khô.
2. Phân tích các ca sai lớn raw/proc: chênh mốc/xử lý nguồn có thể làm delta khó học; không xóa dữ liệu chỉ vì model sai.
3. Thí nghiệm kế tiếp có thể kiểm tra thêm lịch sử dài hơn hoặc đặc trưng bền vững với nhiễu trong một thay đổi riêng; phải bảo đảm chỉ dùng quá khứ và so baseline trên cùng cửa sổ. Chưa thực hiện để tránh tối ưu liên tục trên validation cũ.
4. Xác định tập xác nhận mới trước khi tiếp tục chọn model. Khoảng sai số chỉ được trình bày như khoảng dự báo khi có quy trình hiệu chuẩn/kiểm định phù hợp.

Nguồn: FloodNet NYU/CUNY. Dữ liệu và artifacts giữ cục bộ; tuân giấy phép nguồn.
