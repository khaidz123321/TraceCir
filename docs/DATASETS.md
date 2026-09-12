# Dataset setup

None of the datasets used by the paper can be auto-downloaded (they require
manual agreement / registration steps), so this documents exactly what to
fetch and where to place it. All paths below are what `src/data/datasets.py`
and the scripts under `src/scripts/` expect.

## 1. Pre-training set (for OTI + Phi): ImageNet1K test split

- Source: https://image-net.org/download.php (registration required).
- The paper uses the **unlabeled test split** (100K images) of ILSVRC2012.
- Place the images (flat or nested, they are found recursively) under:
  ```
  data/ImageNet1K/test/
  ```

## 2. FashionIQ

- Images: follow instructions at https://github.com/XiaoxiaoGuo/fashion-iq
  (images are hosted externally; the repo provides a download script/URLs).
- Annotations (captions + splits): https://github.com/XiaoxiaoGuo/fashion-iq
- Expected layout:
  ```
  data/FashionIQ/
      captions/cap.{dress,shirt,toptee}.{train,val,test}.json
      image_splits/split.{dress,shirt,toptee}.{train,val,test}.json
      images/*.jpg
  ```

## 3. CIRR

- Request access / download from https://github.com/Cuberick-Orion/CIRR
- Expected layout:
  ```
  data/CIRR/
      train/, dev/, test1/                       (image folders, as shipped)
      cirr/captions/cap.rc2.{train,val,test1}.json
      cirr/image_splits/split.rc2.{train,val,test1}.json
  ```

## 4. CIRCO (proposed by this paper)

- Annotations + instructions: https://github.com/miccunifi/CIRCO
- Images: COCO 2017 **unlabeled** split, https://cocodataset.org/#download
  (`unlabeled2017.zip`).
- Expected layout:
  ```
  data/CIRCO/
      annotations/{val,test}.json
      COCO2017_unlabeled/unlabeled2017/*.jpg
  ```
- The official CIRCO test-set ground truths are withheld; submit predictions
  to https://circo.micc.unifi.it/ to get test-set metrics (see paper Sec. 4).

## 5. Open Images V7 concept vocabulary (for OTI / Phi's GPT regularization)

- Class names: export the `DisplayName` column of the official
  class-descriptions CSV from
  https://storage.googleapis.com/openimages/web/download_v7.html
  ("Boxes" > "Class Names") into a plain text file, one class per line:
  ```
  data/open_images_v7_classes.txt
  ```

## 6. Pre-generated GPT-Neo phrases

- Run once, offline (see Appendix A — ~12h on a single A100 for the full
  ~20,932-class vocabulary):
  ```bash
  python -c "
  from src.concepts import load_open_images_vocab, pregenerate_gpt_phrases
  vocab = load_open_images_vocab('data/open_images_v7_classes.txt')
  pregenerate_gpt_phrases(vocab, 'data/gpt_phrases.jsonl')
  "
  ```
- For a quick smoke test, generate phrases for a small vocab subset first
  (e.g. the first 50 lines of the class list) to validate the pipeline
  before committing to the full run.
