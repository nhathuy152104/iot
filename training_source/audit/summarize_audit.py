"""Normalize derived labels and write a local Vietnamese audit report."""
import collections
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def read_csv(name):
    with (OUT / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    for name in ('stations.csv', 'station_month_quality.csv', 'candidate_episodes.csv'):
        rows = read_csv(name)
        for r in rows:
            r['type'] = '|'.join(sorted(set(x.strip().lower() for x in r['type'].split('|'))))
        with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    stations = read_csv('stations.csv')
    episodes = read_csv('candidate_episodes.csv')
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    summary['types'] = dict(collections.Counter(r['type'] for r in stations))
    summary['sustained_counts'] = {typ: {str(h): sum(r['type'] == typ and int(r['threshold_mm']) == h
        and r['sustained'] == 'True' for r in episodes) for h in (10, 100, 200)}
        for typ in sorted({r['type'] for r in stations})}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    n = summary['input_rows']
    totals = summary['totals']
    pluvial = [r for r in stations if r['type'] == 'pluvial']
    ranked = sorted(pluvial, key=lambda r: int(r['sustained_episodes_ge_100mm']), reverse=True)
    years = collections.Counter(r['start'][:4] for r in episodes if r['type'] == 'pluvial'
        and r['threshold_mm'] == '100' and r['sustained'] == 'True')
    by_type = {}
    for typ in summary['types']:
        ss = [r for r in stations if r['type'] == typ]
        by_type[typ] = {key: sum(int(r[key]) for r in ss) for key in ('rows', 'missing_proc', 'samples_ge_100mm', 'gaps_gt_10min')}
    def pct(x, base=n):
        return f'{100*x/base:.3f}%'
    lines = [
        '# Kiểm kê chất lượng và ứng viên ngập FloodNet', '',
        '## Kết luận', '',
        'Dữ liệu đủ quy mô để bắt đầu thử nghiệm baseline và XGBoost. Chưa chứng minh model dự báo tốt hoặc áp dụng được tại Việt Nam. Chưa train model.', '',
        '## Phạm vi quét', '',
        f'- {summary["files"]} CSV; {summary["bytes"]:,} byte; {n:,} bản ghi; {summary["deployments"]} deployment_id.',
        '- Các file từ 10/2020 đến 10/2023; thiếu file 09/2023. Không khẳng định mỗi trạm liên tục đủ ba năm.',
        f'- Loại triển khai sau chuẩn hóa khoảng trắng: {summary["types"]}.',
        f'- Thời gian không hợp lệ: {summary["invalid_time"]}; đảo thời gian trong từng trạm/file: {summary["input_time_reversals"]}.',
        f'- Trùng thời điểm: {totals.get("duplicate_timestamps", 0)}; trùng có xung đột: {totals.get("conflicting_duplicate_timestamps", 0)}.', '',
        '## Chất lượng dữ liệu', '',
        f'- Thiếu/không hữu hạn depth_proc_mm: {totals["missing_proc"]:,} ({pct(totals["missing_proc"])} tổng số hàng).',
        f'- Khoảng giữa hai mẫu liên tiếp trên 10 phút: {totals["gaps_gt_10min"]:,}. Đây là số khoảng, không phải số hàng mất hoặc thời lượng mất.',
        f'- depth_raw_mm âm: {totals["negative_raw"]:,} ({pct(totals["negative_raw"])}). Không tự coi là lỗi: mốc nền và nhiễu có thể tạo độ sâu thô âm.',
        f'- Chênh lệch tuyệt đối raw/proc >=100 mm: {totals["raw_proc_difference_ge_100mm"]:,} hàng. Không chứng minh bộ lọc sai.',
        f'- Raw >=100 mm nhưng proc <10 mm: {totals["raw_ge_100mm_proc_lt_10mm"]:,} hàng; cần xem xét các cửa sổ liên quan trước chọn đặc trưng.',
        f'- Chuyển tiếp proc có độ dốc tuyệt đối >254 mm/phút trong khoảng liên tục: {totals["proc_abs_gradient_gt_254mm_per_min"]:,}; cờ đối soát, không tự xóa.',
        '- Không có kiểm định ngoại cảnh; chưa xác minh bộ lọc archive có sử dụng dữ liệu tương lai.', '',
        '## Đợt ứng viên có duy trì', '',
        'Điều kiện: >=3 mẫu khác thời điểm, kéo dài >=2 phút; tách khi xuống dưới ngưỡng, mẫu proc thiếu hoặc khoảng cách >max(3 phút, 3 lần chu kỳ trung vị tháng).', '',
        '| Loại | >=1 cm | >=10 cm | >=20 cm |', '|---|---:|---:|---:|']
    for typ, counts in summary['sustained_counts'].items():
        lines.append(f'| {typ} | {counts["10"]:,} | {counts["100"]:,} | {counts["200"]:,} |')
    lines += ['', 'Ba mức lồng nhau, không cộng lại thành tổng sự kiện. Một đợt ngập thật có thể bị chia thành nhiều đoạn do nhiễu, mất mẫu hoặc dao động quanh ngưỡng.',
        f'Các đoạn pluvial >=10 cm có duy trì xuất hiện theo năm bắt đầu: {dict(sorted(years.items()))}.',
        f'Nhóm theo thời gian cách nhau <=6 giờ tạo {summary["pluvial_100mm_temporal_clusters_6h"]} cụm; KHÔNG phải số trận mưa độc lập đã xác minh.', '',
        '## Các trạm cần xem trước', '',
        'Xếp theo số đoạn >=10 cm có duy trì, không phải xếp hạng chất lượng. Mẫu thiếu chỉ tính trên các hàng có mặt, không bao gồm mọi thời gian offline.', '',
        '| deployment_id | Trạm | Số hàng | Chu kỳ tháng trung vị (s) | Thiếu proc | Đoạn >=10 cm | Max proc (cm) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in ranked[:12]:
        lines.append(f'| {r["deployment_id"]} | {r["name"]} | {int(r["rows"]):,} | {r["median_of_monthly_cadences_sec"]} | {pct(int(r["missing_proc"]), int(r["rows"]))} | {r["sustained_episodes_ge_100mm"]} | {r["max_proc_cm"]} |')
    top4 = sum(int(r['sustained_episodes_ge_100mm']) for r in ranked[:4])
    all_pluvial = summary['sustained_counts']['pluvial']['100']
    lines += ['', '## Cảnh báo khi lựa chọn trạm', '',
        f'- Bốn trạm đứng đầu chiếm {top4:,}/{all_pluvial:,} đoạn pluvial >=10 cm ({pct(top4, all_pluvial)}). Số lượng lớn tập trung không chứng minh chất lượng tốt; cần đối soát noise và độ phân mảnh.',
        '- daily_new_falcon là ứng viên đáng xem trước: chuỗi dài, 115 đoạn >=10 cm có duy trì, chỉ 40 hàng thiếu proc trên 1.029.482 hàng, không có chuyển tiếp proc vượt 254 mm/phút theo phép quét. Đây là lựa chọn sơ bộ, không xác nhận mọi đoạn là ngập thật.',
        '- Các trạm như simply_pet_joey và future_aqua_wildcat có nhiều đoạn >=10 cm đồng thời >=20 cm, có các đỉnh lớn và cờ độ dốc. Cần xem đồ thị, không tự loại hoặc coi tất cả là sự kiện ngập.', '',
        '## Bước tiếp theo trước train', '',
        '1. Xem đồ thị các cửa sổ ngập và raw/filt/proc ở nhóm pluvial; rà các đỉnh và bước nhảy, kể cả trạm có nhiều đoạn.',
        '2. Chọn nhóm trạm có chu kỳ tương thích. Không nội suy từ 5 phút thành 1 phút rồi coi là quan trắc thật.',
        '3. Xác minh bộ lọc causal/độ trễ; chỉ dùng đầu vào thực sự sẵn có tại thời điểm dự báo. Nhãn đã đối soát có thể là sản phẩm hậu kỳ.',
        '4. Đếm số cửa sổ 10 phút hợp lệ và nhãn tương lai 10 phút sau khi xử lý khoảng thiếu; audit này chưa đếm cửa sổ train.',
        '5. Chia thời gian/sự kiện và giữ tập test; purge cửa sổ chồng qua ranh giới; nhóm nhiều trạm cùng sự kiện để tránh rò rỉ.',
        '6. So sánh baseline và XGBoost, riêng trên lúc có ngập và lúc chưa ngập trước khởi phát; không chỉ báo cáo MAE toàn bộ ngày khô.', '',
        '## Đầu ra và giấy phép', '',
        'Các bảng chi tiết nằm cạnh báo cáo: summary.json, files.csv, stations.csv, station_month_quality.csv, candidate_episodes.csv, temporal_clusters.csv. Không sửa CSV nguồn.',
        'Nguồn: FloodNet (New York University and The City University of New York). Chỉ dùng phi thương mại theo thỏa thuận được cung cấp. Không phân phối dữ liệu gốc; cần rà quyền trước công khai các đầu ra dẫn xuất. Chưa xác minh ngày tải/thỏa thuận tương ứng của từng file; người tải cần bổ sung thông tin này.',
        'Phân tích/biểu diễn phái sinh được phép: CC BY-NC-SA 4.0, kèm các điều kiện bổ sung của Data Access License Agreement.', '']
    (OUT / 'BAO_CAO_KIEM_KE.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'by_type': by_type, 'pluvial_episodes_by_start_year': dict(years),
                      'pluvial_stations_with_sustained_100mm': sum(int(r['sustained_episodes_ge_100mm']) > 0 for r in pluvial),
                      'top_stations': ranked[:6]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
