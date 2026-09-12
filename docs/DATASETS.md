# Thiết lập dữ liệu

Không có bộ dữ liệu nào ở đây tự động tải về được (đều cần bước đăng ký /
xin quyền thủ công), nên tài liệu này ghi rõ chính xác cần tải gì và đặt ở
đâu. Toàn bộ đường dẫn bên dưới là những gì `src/data/datasets.py` và các
script trong `src/scripts/` yêu cầu.

## 1. Tập dữ liệu pre-training (dùng cho OTI + Phi): ImageNet1K test split

- Nguồn: https://image-net.org/download.php (cần đăng ký tài khoản).
- Bài báo dùng **tập test không nhãn** (100K ảnh) của ILSVRC2012.
- Đặt ảnh (dạng phẳng hoặc lồng thư mục đều được, code sẽ tự quét đệ quy)
  vào:
  ```
  data/ImageNet1K/test/
  ```

## 2. FashionIQ

- Ảnh: làm theo hướng dẫn tại https://github.com/XiaoxiaoGuo/fashion-iq
  (ảnh được host ở nơi khác; repo này cung cấp script/link tải).
- Annotation (câu mô tả + chia tập): https://github.com/XiaoxiaoGuo/fashion-iq
- Cấu trúc thư mục cần có:
  ```
  data/FashionIQ/
      captions/cap.{dress,shirt,toptee}.{train,val,test}.json
      image_splits/split.{dress,shirt,toptee}.{train,val,test}.json
      images/*.jpg
  ```

## 3. CIRR

Theo README chính thức tại https://github.com/Cuberick-Orion/CIRR, dữ liệu có 2 phần tách biệt: **annotation** (tải tự do) và **ảnh gốc** (cần xin quyền qua NLVR2).

### 3.1. Annotation (không cần xin quyền)

```bash
mkdir -p data/CIRR
cd data/CIRR
git clone -b cirr_dataset https://github.com/Cuberick-Orion/CIRR.git cirr
```

### 3.2. Ảnh gốc (cần xin quyền — ảnh gốc thuộc về NLVR2, không phải CIRR)

1. Điền Google Form đồng ý điều khoản của nhóm NLVR2, hướng dẫn tại:
   https://github.com/lil-lab/nlvr/tree/master/nlvr2#direct-image-download
2. Nếu nhóm NLVR2 không phản hồi, email tác giả CIRR (zheyuan.david.liu@outlook.com)
   và nêu rõ bạn đã điền form của NLVR2 đồng ý điều khoản của họ.
3. Sau khi nhận được ảnh, đặt vào đúng cấu trúc `img_raw/` bên dưới.

**Không cần** tải `img_feat_res152/` hay `img_feat_frcnn/` (pre-extracted features) — code này tự trích đặc trưng bằng CLIP, không dùng ResNet152/F-RCNN.

### Cấu trúc thư mục cần có

```
data/CIRR/
    cirr/
        captions/cap.rc2.{train,val,test1}.json
        image_splits/split.rc2.{train,val,test1}.json
        img_raw/
            train/<0-99>/<image>.png     (thư mục con kế thừa từ NLVR2, không mang ý nghĩa gì đặc biệt)
            dev/<image>.png
            test1/<image>.png
```

`--data-root` truyền cho `scripts/validate.py`/dataset loader trỏ vào `data/CIRR` (thư mục cha chứa `cirr/`).

## 4. CIRCO (do chính bài báo này đề xuất)

- Annotation + hướng dẫn: https://github.com/miccunifi/CIRCO
- Ảnh: tập COCO 2017 **unlabeled**, https://cocodataset.org/#download
  (file `unlabeled2017.zip`).
- Cấu trúc thư mục cần có:
  ```
  data/CIRCO/
      annotations/{val,test}.json
      COCO2017_unlabeled/unlabeled2017/*.jpg
  ```
- Nhãn ground truth của tập test CIRCO không được công bố công khai; cần
  nộp kết quả dự đoán lên https://circo.micc.unifi.it/ để lấy số liệu trên
  tập test (xem Mục 4 của bài báo).

## 5. Từ vựng concept Open Images V7 (dùng cho loss GPT-regularization của OTI/Phi)

- Tên các lớp (class name): xuất cột `DisplayName` từ file CSV
  class-descriptions chính thức tại
  https://storage.googleapis.com/openimages/web/download_v7.html
  (mục "Boxes" > "Class Names") ra một file text thuần, mỗi lớp 1 dòng:
  ```
  data/open_images_v7_classes.txt
  ```

## 6. Các phrase GPT-Neo được sinh sẵn (pre-generated)

- Chạy 1 lần duy nhất, offline (xem Phụ lục A — mất khoảng 12 giờ trên 1
  GPU A100 cho toàn bộ từ vựng ~20.932 lớp):
  ```bash
  python -c "
  from src.concepts import load_open_images_vocab, pregenerate_gpt_phrases
  vocab = load_open_images_vocab('data/open_images_v7_classes.txt')
  pregenerate_gpt_phrases(vocab, 'data/gpt_phrases.jsonl')
  "
  ```
- Để kiểm tra nhanh (smoke test), hãy sinh phrase cho một tập con nhỏ của
  từ vựng trước (ví dụ 50 dòng đầu của danh sách lớp) để xác nhận pipeline
  chạy đúng trước khi chạy toàn bộ.
