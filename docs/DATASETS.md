# Thiết lập dữ liệu (P0: CIRCO + CIRR)

P0 chỉ dùng **CIRCO validation** và **CIRR validation** (xem
`P0_TRACE_CIR_Experimental_QKhai.docx`, mục "Out of scope: FashionIQ").
Không cần ImageNet1K/Open Images/GPT phrases nữa — đó là dữ liệu
pre-training riêng cho SEARLE (đã tạm dừng), P0 không train gì cả.

## 1. CIRR

Theo README chính thức tại https://github.com/Cuberick-Orion/CIRR, dữ liệu có 2 phần tách biệt: **annotation** (tải tự do) và **ảnh gốc** (cần xin quyền qua NLVR2).

### 1.1. Annotation (không cần xin quyền)

```bash
mkdir -p data/CIRR
cd data/CIRR
git clone -b cirr_dataset https://github.com/Cuberick-Orion/CIRR.git cirr
```

### 1.2. Ảnh gốc (cần xin quyền — ảnh gốc thuộc về NLVR2, không phải CIRR)

1. Điền Google Form đồng ý điều khoản của nhóm NLVR2, hướng dẫn tại:
   https://github.com/lil-lab/nlvr/tree/master/nlvr2#direct-image-download
2. Nếu nhóm NLVR2 không phản hồi, email tác giả CIRR (zheyuan.david.liu@outlook.com)
   và nêu rõ bạn đã điền form của NLVR2 đồng ý điều khoản của họ.
3. Sau khi nhận được ảnh, đặt vào đúng cấu trúc `img_raw/` bên dưới.

**Không cần** tải `img_feat_res152/` hay `img_feat_frcnn/` (pre-extracted features) — code này tự trích đặc trưng bằng OpenCLIP, không dùng ResNet152/F-RCNN.

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

`--data-root` trỏ vào `data/CIRR` (thư mục cha chứa `cirr/`).

## 2. CIRCO

- Annotation + hướng dẫn: https://github.com/miccunifi/CIRCO
- Ảnh: tập COCO 2017 **unlabeled**, https://cocodataset.org/#download
  (file `unlabeled2017.zip`, ~20GB, 123,403 ảnh).
- Cấu trúc thư mục cần có:
  ```
  data/CIRCO/
      annotations/{val,test}.json
      COCO2017_unlabeled/unlabeled2017/*.jpg
  ```
- Nhãn ground truth của tập **test** không được công bố công khai — P0 chỉ
  dùng tập **validation** nên không bị ảnh hưởng bởi giới hạn này.

## Trạng thái hiện tại (repo này)

| Bộ dữ liệu | Trạng thái |
|---|---|
| CIRCO | ✅ Đầy đủ (`data/circo/circo/`) |
| CIRR | ⏳ Chờ NLVR2 duyệt form |
