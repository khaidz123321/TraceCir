# P0: TRACE-CIR — Source-Grounded Local Transition Matching

Thực nghiệm P0 theo đúng giao thức trong `P0_TRACE_CIR_Experimental_QKhai.docx`:
kiểm chứng 3 giả thuyết (H1/H2/H3) của phương pháp mới **TRACE-CIR**, xây
dựng trên nền **TAPR** ("Atomic Edit Factorization for Training-Free
Composed Image Retrieval") — không train, dùng OpenCLIP đóng băng + 1
compiler đa phương thức (Qwen2.5-VL-7B-Instruct) để phân rã câu lệnh chỉnh
sửa thành các atom ADD/REMOVE/PRESERVE/REPLACE.

> **Lưu ý:** Repo này trước đó dựng baseline cho bài **SEARLE** (Zero-Shot
> Composed Image Retrieval with Textual Inversion, arXiv:2303.15247). Công
> việc đó đã **tạm dừng** để tập trung vào P0 — lịch sử code vẫn còn trong
> git log nếu cần dùng lại (ví dụ để so sánh trong phần liên quan của báo
> cáo/paper), nhưng không phải trọng tâm hiện tại.

## Cấu trúc thư mục

```
src/
  models/
    openclip_utils.py   Load OpenCLIP ViT-L/14 + trích global/local vector (Eq. 3 TAPR)
  data/
    datasets.py          CIRRDataset, CIRCODataset (annotation + ảnh)
  p0/
    scoring.py            E0: chấm điểm kiểu TAPR (Eq. 6-10) — Target/Add/Preserve/Remove
    compiler.py            Transition Compiler: Qwen2.5-VL-7B -> {target, atoms[]}
    feature_cache.py       Trích + cache đặc trưng global/local dùng chung cho E0-E4
    run_e0.py               CLI: chạy E0 trên CIRCO/CIRR validation, ra Recall@K/mAP@K
  eval.py                  Recall@K, RecallSubset@K, mAP@K (dùng lại được cho cả P0)
  seed.py                  Cố định random seed
```

## Quy trình chạy E0 (baseline TAPR)

1. **Cache đặc trưng** cho toàn bộ ảnh trong tập classic/index (CIRCO hoặc CIRR):
   ```python
   from src.models.openclip_utils import load_openclip
   from src.data.datasets import CIRCODataset
   from src.p0.feature_cache import build_feature_cache

   model, preprocess, _ = load_openclip(device="cuda")
   ds = CIRCODataset("data/CIRCO", "val", "classic", preprocess)
   build_feature_cache(model, ds, "features/circo", id_key="image_id", device="cuda")
   ```
2. **Chạy Transition Compiler** trên tập query (relative split) — cần GPU đủ mạnh
   (khuyến nghị Kaggle, không chạy được trên máy không GPU):
   ```python
   from src.p0.compiler import load_compiler, run_compiler_batch
   model, processor = load_compiler()
   run_compiler_batch(model, processor, queries, "data/circo_compiled.jsonl")
   ```
3. **Audit thủ công** (bắt buộc, mục 5.3 giao thức): kiểm tra ~100 CIRCO + ~100 CIRR
   output của compiler, gán nhãn Correct/Partially/Incorrect. Nếu tỉ lệ Correct
   < ~85%, sửa lại prompt trước khi chạy full.
4. **Chạy E0**:
   ```bash
   python -m src.p0.run_e0 \
       --dataset circo --split val --data-root data/CIRCO \
       --feature-cache-dir features/circo \
       --compiled-queries-path data/circo_compiled.jsonl
   ```

## Trạng thái dữ liệu

| Bộ dữ liệu | Trạng thái |
|---|---|
| CIRCO (annotation + 123,403 ảnh COCO) | ✅ Đầy đủ |
| CIRR validation | ⏳ Chờ NLVR2 duyệt |

## Đã kiểm chứng (chưa chạy được compiler thật vì môi trường dev không có GPU)

- OpenCLIP ViT-L/14 load + trích global (768-d, L2-norm=1) + local (64×768-d,
  L2-norm=1) đúng công thức, test trên ảnh CIRCO thật.
- Công thức chấm điểm E0 (`scoring.py`) phản ứng đúng hướng: thêm bằng chứng
  Add → điểm tăng; thêm bằng chứng Remove → điểm giảm; ảnh giống hệt tham
  chiếu → điểm Preserve (continuity) tối đa.
- Toàn bộ pipeline cache → scoring → eval chạy end-to-end không lỗi trên
  subset ảnh CIRCO thật (compiler được giả lập bằng output mẫu).
- **Chưa kiểm chứng**: compiler thật (Qwen2.5-VL-7B) — cần chạy trên máy có
  GPU (dự kiến Kaggle) trước khi tin tưởng số liệu E0 cuối cùng.

## Khoảng trống cần lưu ý

- Bài báo TAPR không công bố giá trị mặc định cụ thể cho trọng số
  θ_T/θ_A/θ_P/θ_R (Eq. 10) trong phần văn bản trích xuất được — `E0Weights`
  hiện dùng trọng số bằng nhau (1.0 mỗi cái) làm mặc định tạm thời, cần coi
  đây là siêu tham số cần tinh chỉnh/báo cáo riêng, không phải số liệu gốc
  từ bài báo.
