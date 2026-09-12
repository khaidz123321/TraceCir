# Dựng lại baseline SEARLE

Bản tái hiện (reproduction) baseline của bài báo **"Zero-Shot Composed Image
Retrieval with Textual Inversion"** (Baldrati, Agnolucci, Bertini, Del Bimbo
— ICCV 2023, [arXiv:2303.15247](https://arxiv.org/abs/2303.15247)), được
xây dựng lại từ đầu dựa theo các công thức và kiến trúc trong bài báo, có
đối chiếu với cấu trúc repo chính thức
([miccunifi/SEARLE](https://github.com/miccunifi/SEARLE)) để bám sát quy
ước về thư mục/tham số dòng lệnh (CLI).

Phương pháp của bài báo (**SEARLE**) biến bài toán Composed Image Retrieval
(ảnh + câu mô tả tương đối -> ảnh đích) thành bài toán quen thuộc là
text-to-image retrieval, bằng cách ánh xạ ảnh tham chiếu thành một token
"từ giả" (pseudo-word) rồi chèn token này vào câu mô tả trước khi đưa qua
bộ mã hóa văn bản (text tower) của CLIP.

## Cấu trúc thư mục

```
src/
  models/
    phi.py          Mạng Textual Inversion Network (Bảng 5): MLP 3 lớp
    clip_utils.py   Load CLIP + encode_with_pseudo_tokens (chèn token vào chuỗi)
  data/
    datasets.py     Dataset loader cho FashionIQ / CIRR / CIRCO / ảnh không nhãn
  losses.py         L_cos, L_gpt, L_distil (Eq. 1, 2, 4)
  concepts.py       Gán concept bằng CLIP zero-shot + cache phrase từ GPT-Neo
  oti.py            Tầng 1: Optimization-based Textual Inversion (Mục 3.1)
  train_phi.py      Tầng 2: distill nhãn giả OTI sang mạng Phi (Mục 3.2)
  baselines.py      Baseline không cần train: Image-only / Text-only / Image+Text
  eval.py           Recall@K, mAP@K (Eq. 6)
  scripts/
    run_oti.py      CLI: chạy tầng 1 trên toàn bộ tập ảnh pre-training
    validate.py     CLI: tầng 3, suy luận + đánh giá zero-shot CIR
docs/
  DATASETS.md       Hướng dẫn lấy FashionIQ / CIRR / CIRCO / ImageNet / từ vựng concept
```

## Quy trình (Pipeline)

1. **Chuẩn bị dữ liệu** — xem [docs/DATASETS.md](docs/DATASETS.md). Không
   bộ dữ liệu nào trong số này có thể tự động tải về (đều cần đăng ký/xin
   quyền thủ công).
2. **Tầng 1 — OTI** (tạo ra các "nhãn giả" pseudo-word, chạy 1 lần, khá chậm):
   ```bash
   python -m src.scripts.run_oti \
       --image-dir data/ImageNet1K/test \
       --vocab-path data/open_images_v7_classes.txt \
       --gpt-phrases-path data/gpt_phrases.jsonl \
       --output-path data/oti_targets.pt \
       --clip-model-name ViT-B/32
   ```
3. **Tầng 2 — huấn luyện Phi** (distillation, Mục 3.2):
   ```bash
   python -m src.train_phi \
       --image-dir data/ImageNet1K/test \
       --oti-targets-path data/oti_targets.pt \
       --gpt-phrases-path data/gpt_phrases.jsonl \
       --concepts-path data/concepts.json \
       --output-dir checkpoints/phi_b32 \
       --clip-model-name ViT-B/32
   ```
4. **Tầng 3 — đánh giá zero-shot CIR**:
   ```bash
   python -m src.scripts.validate \
       --dataset cirr --split val --data-root data/CIRR \
       --phi-checkpoint checkpoints/phi_b32/phi_final.pt \
       --clip-model-name ViT-B/32
   ```

## Siêu tham số tham khảo (Phụ lục A của bài báo)

| Tầng | Thiết lập | Giá trị |
|---|---|---|
| OTI | số bước lặp | 350 |
| OTI | learning rate | 2e-2 |
| OTI | lambda_cos / lambda_gpt | 1.0 / 0.5 |
| OTI | hệ số EMA | 0.99 |
| OTI | số concept mỗi ảnh (k) | 15 |
| Phi | số epoch (B/32, L/14) | 100, 50 |
| Phi | learning rate | 1e-4 |
| Phi | batch size | 256 |
| Phi | lambda_distil / lambda_gpt | 1.0 / 0.75 |
| Phi | nhiệt độ (temperature) distillation | 0.25 |
| Phi | số concept mỗi ảnh (k) | 150 |
| Cả hai | optimizer | AdamW, weight decay 0.01 |

## Ghi chú về phạm vi / trạng thái hiện tại

Đây là bản **tái hiện baseline dựng lại từ đầu**, không phải codebase chính
thức. Một số điểm đơn giản hóa cần lưu ý trước khi tin tưởng hoàn toàn vào
các con số so với Bảng 1-3 của bài báo:

- `oti.py` xử lý theo batch và dùng chung 1 lượt chọn ngẫu nhiên
  template/GPT-phrase cho mỗi bước tối ưu trên cả batch, thay vì lịch trình
  chính xác theo từng ảnh riêng lẻ như repo chính thức; hành vi hội tụ có
  thể khác đôi chút so với số liệu báo cáo trong bài.
- Phần đánh giá mAP cho CIRCO trong `scripts/validate.py` cần ánh xạ
  `gt_img_ids` của từng câu truy vấn sang vị trí trong tập index; bạn cần
  nối logic này khớp với định dạng annotation CIRCO thực tế của mình trước
  khi tin vào số liệu mAP (xem ghi chú được in ra khi chạy script và
  `docs/DATASETS.md`).
- Chưa bao gồm định dạng nộp bài (submission) cho evaluation server của
  tập test CIRCO; xem https://circo.micc.unifi.it/ để biết thêm.
