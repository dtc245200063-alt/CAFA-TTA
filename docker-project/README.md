# CAFA-TTA — Training bằng Docker (chạy thủ công)

Bộ này chỉ cung cấp môi trường Docker và lệnh để **bạn tự bắt đầu train**. Không có training tự động và không ghi đè checkpoint nghiên cứu. Mã gốc `/project` được mount **read-only**. Mỗi lượt train được lưu ở `docker_runs/<run-name>/workspace/` trên máy.

## Yêu cầu

- Windows 11 + Docker Desktop đang chạy, sử dụng **WSL 2 backend**.
- NVIDIA driver mới, hỗ trợ GPU trong container; GPU RTX 3050 có 4 GB VRAM nên cần quan sát bộ nhớ.
- Dự án gốc `C:\AI-Projects\CAFA-TTA` có `scripts/`, `checkpoints/` và `data/raw/medmnist/pathmnist.npz`.
- Kết nối mạng để build image và tải thư viện (có thể tải nhiều GB).

Nếu dự án gốc ở vị trí khác, sao chép `.env.example` thành `.env` và chỉnh `CAFA_PROJECT_PATH`.

## Bước 1 — Mở PowerShell tại thư mục đã giải nén

```powershell
docker version
docker compose version
```

## Bước 2 — Build môi trường (không train)

```powershell
docker compose build
```

## Bước 3 — Kiểm tra GPU (không train)

```powershell
docker compose run --rm trainer --check-gpu
```

Kết quả cần có `CUDA available: True`.

## Bước 4 — TỰ QUYẾT ĐỊNH bắt đầu train

Chạy thử **baseline 1 epoch**; lưu vào thư mục riêng tên `baseline_try_01`:

```powershell
docker compose run --rm trainer --model baseline --epochs 1 --run-name baseline_try_01
```

Để chạy baseline theo cấu hình gốc **3 epoch**:

```powershell
docker compose run --rm trainer --model baseline --run-name baseline_full_01
```

Enhanced từ checkpoint baseline gốc; số epoch mặc định 15:

```powershell
docker compose run --rm trainer --model enhanced --run-name enhanced_full_01 --from-run baseline_full_01
```

Robust và Balanced từ checkpoint Enhanced gốc; cấu hình mặc định 8 và 6 epoch:

```powershell
docker compose run --rm trainer --model robust --run-name robust_full_01 --from-run enhanced_full_01
docker compose run --rm trainer --model balanced --run-name balanced_full_01 --from-run enhanced_full_01
```

**Không dùng lại run-name cũ.** Chương trình chủ động từ chối ghi đè. Các checkpoint sau train lưu trong `docker_runs/<run-name>/workspace/checkpoints/`; lịch sử Enhanced/Robust/Balanced lưu trong thư mục `experiments` của workspace tương ứng.

## Lưu ý quan trọng

- Tùy chọn `--epochs 1` chỉ đổi `EPOCHS` trong **bản sao** của script, phục vụ smoke test, **không phải kết quả nghiên cứu gốc**.
- Không tự động train khi bạn build hoặc kiểm tra GPU. Training chỉ xảy ra khi bạn nhập `--model ... --run-name ...`.
- Docker GPU trên Windows cần WSL 2; Docker Desktop/driver/WSL có thể cần cấu hình thêm.
- Chưa xác thực build/run trong môi trường Docker Desktop/RTX 3050 của máy bạn. Hãy kiểm tra GPU và chạy 1 epoch trước.
- Docker package dùng tập thư viện tối thiểu cho bốn script, không thay thế toàn bộ `requirements.txt` hoặc quy trình reproducibility nghiên cứu.
- Giao diện Windows Tkinter hiện tại vẫn chạy bên ngoài Docker; bộ này **chỉ phục vụ huấn luyện**.
- Không đưa thư mục `docker_runs` lên GitHub; có thể có checkpoint, nhãn và dữ liệu thực nghiệm.

## Ghi chú về phạm vi (cập nhật)

- `CAFA_PROJECT_PATH` trong `.env` hiện được `compose.yaml` / `compose.cpu.yaml` sử dụng (mặc định `..`).
- For `enhanced`, `robust`, and `balanced`, use `--from-run` to reuse the prerequisite checkpoint from an earlier run in `docker_runs/`. Without `--from-run`, the prerequisite checkpoint must exist in `/project/checkpoints/`. Legacy checkpoints are not included in Git.
- Bộ Docker này không bao gồm mô hình 17 lớp; xem `scripts/train_unified.py` và README chính.
- Split Kather trong `dataset_splits/kather2016_seed42.json` (3496/752/752) khác split của mô hình 17 lớp (3496/744/760).

## Reusing checkpoints between Docker runs

The original PathMNIST training modes use separate run workspaces.
A fresh clone does not include the original baseline/enhanced checkpoints.

Run the stages in order:

1. `baseline` produces `resnet18_pathmnist_best.pt`.
2. `enhanced --from-run baseline_01` uses that checkpoint.
3. `robust --from-run enhanced_01` or
   `balanced --from-run enhanced_01` uses
   `resnet18_pathmnist_enhanced_best.pt`.

Example commands (Ubuntu, from `docker-project/`, GPU required):

```bash
docker compose run --rm trainer --model baseline --run-name baseline_01
docker compose run --rm trainer --model enhanced --run-name enhanced_01 --from-run baseline_01
docker compose run --rm trainer --model robust --run-name robust_01 --from-run enhanced_01
```

The `--from-run` option reads the prerequisite checkpoint from
`docker_runs/<previous-run>/workspace/checkpoints/`.
Without it, the launcher still checks `/project/checkpoints/`.

Each run name must be unique. Existing run directories are not overwritten.
Source code and original checkpoints remain read-only.
The required PathMNIST data must be prepared before training.

These are the original PathMNIST training modes, not the unified
17-class ResNet18 pipeline.
