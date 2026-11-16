# Lượt không hoàn tất — chưa fit

Dừng trước fit vì tham số mặc định `missing=NaN` của XGBoost không thể ghi JSON chuẩn với `allow_nan=False`. Đã chuyển sang ghi rõ chính sách NaN dạng chuỗi, không đổi hành vi model.

Run thành công: `../20261111T010522_872558Z_depth_mae_trial/`. Không dùng lượt này làm kết quả thí nghiệm.
