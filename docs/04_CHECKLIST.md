# 04 — Checklist nghiệm thu bàn giao

## Người giao

- [ ] Kiểm tra quyền gửi dữ liệu/model với người nhận; chưa rõ thì chưa phát tán.
- [ ] Gửi folder này nguyên cấu trúc và link tải nguồn; không kèm CSV/cache/cửa sổ quan trắc.
- [ ] Chỉ rõ model hiện dùng là raw ba mã, không phải model mười mã/trial MAE.
- [ ] Giữ bản run gốc và checksum; không gửi `.venv`, secrets hoặc file không liên quan.

## Người nhận chạy trên máy của mình

- [ ] Đọc README và tài liệu 01–03.
- [ ] Dựng môi trường mới; ghi phiên bản Python/thư viện.
- [ ] `verify_manifest.py` PASS.
- [ ] Bộ test trong `tests/` PASS.
- [ ] Chạy JSON mẫu giả, đối chiếu `expected_synthetic.json`; dữ liệu CSV nguồn tự tải từ link.
- [ ] Thử một CSV có lịch sử thiếu; nhận `unavailable`, không phải dự báo được fill.
- [ ] Giải thích được mm/cm, bin UTC, 12 feature, delta và nhãn proc tương lai.
- [ ] Nếu có dữ liệu đầy đủ: kiểm tra runtime với validation gốc bằng `verify_against_project.py`.
- [ ] Nếu train tiếp: tạo run mới, giữ test/tập xác nhận độc lập và ghi quyết định trước fit.
- [ ] Nếu chỉ có CSV tự tải: chạy `train_from_download.py --prepare-only`, đối soát số hàng/nguồn rồi mới chạy fit; không coi kiểm tra trên máy người giao là thay thế nghiệm thu của mình.
- [ ] Biết inference chạy được không đồng nghĩa đã kiểm định trên ESP32 hay đủ an toàn vận hành.

## Biên bản nên ghi thêm

Ngày nhận, tên người nhận, máy/hệ điều hành, phiên bản môi trường, checksum model, các lệnh đã chạy, PASS/FAIL, dữ liệu được phép truy cập và hạng mục còn thiếu. `KIEM_TRA_BAN_GIAO.json` chỉ ghi kiểm tra trên máy chuẩn bị, không thay cho việc người nhận chạy lại trên máy họ.
