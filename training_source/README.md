# Snapshot code nghiên cứu

Các file Python được copy nguyên nội dung từ `analysis/` và `audit/` của project khi bàn giao; không copy `.venv`, dữ liệu, cache, cửa sổ hoặc secrets. Giữ snapshot để hiểu/review phương pháp và khôi phục cây source nếu cần.

**Không đồng nghĩa các entry point lịch sử chạy được chỉ với folder này:** chúng có thể phụ thuộc những run/provenance/cache cố định đã có trong project gốc. Để train một cấu hình raw ba mã từ CSV tự tải mà không cần cache cũ, dùng `../tools/train_from_download.py`; để tái lập đầy đủ lịch sử, xem `../docs/03_DU_LIEU_VA_TRAIN_TIEP.md`.
