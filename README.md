# 🌿 Nghiên cứu và Triển khai Mô hình Phân vùng Ảnh sử dụng Học sâu trên Raspberry Pi

> **Báo cáo Nghiên cứu Khoa học** — Trường Đại học Thăng Long, Khoa Công nghệ Thông tin
>
> Nghiên cứu và đánh giá mô hình **Semantic Segmentation** nhẹ (Fast-SCNN & ENet) cho bài toán phân vùng bệnh cây trồng trên **Raspberry Pi 4**. Kết quả cho thấy **Fast-SCNN** là mô hình tối ưu nhất để triển khai trên thiết bị edge, với hiệu suất vượt trội so với ENet về cả độ chính xác lẫn tốc độ.

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Yêu cầu môi trường](#-yêu-cầu-môi-trường)
- [Cài đặt](#-cài-đặt)
- [Dataset](#-dataset)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Kết quả](#-kết-quả)
- [Tác giả](#-tác-giả)

---

## 📖 Giới thiệu

Dự án này nghiên cứu và triển khai hai mô hình **lightweight semantic segmentation** cho bài toán **phân vùng nhị phân (binary segmentation)** bệnh cây trồng trên **Raspberry Pi 4**, nhằm tìm ra mô hình học sâu tốt nhất phù hợp với thiết bị edge có tài nguyên hạn chế.

Hai mô hình được đánh giá:

| Mô hình | Vai trò | Đặc điểm |
|---------|---------|-----------|
| **Fast-SCNN** ⭐ | **Mô hình chính (đề xuất)** | Learning to Downsample + Feature Fusion + Pyramid Pooling — cho kết quả tốt nhất trên Raspberry Pi 4 |
| **ENet** | Mô hình so sánh | Encoder-Decoder hiệu quả với Bottleneck blocks + PReLU |

### 🏆 Kết luận nghiên cứu

> **Fast-SCNN** được chọn là mô hình tốt nhất cho triển khai trên Raspberry Pi 4, với **Binary Accuracy cao hơn (86.32% vs 83.29%)**, **Dice và IoU vượt trội**, và đặc biệt **tốc độ gấp ~3 lần** so với ENet (23.34 FPS vs 7.94 FPS). ENet chỉ được sử dụng thêm để làm cơ sở so sánh, đánh giá hiệu quả của Fast-SCNN.

### Tính năng chính

- ✅ Huấn luyện với **BCE + Dice Loss** kết hợp
- ✅ Đánh giá bằng **Dice Coefficient** và **IoU Score**
- ✅ Data augmentation (flip, brightness)
- ✅ Hỗ trợ inference đơn ảnh hoặc thư mục
- ✅ Xuất overlay trực quan (ảnh gốc + mask dự đoán)
- ✅ Chuyển đổi sang **TFLite INT8** để chạy trực tiếp trên Raspberry Pi
- ✅ Cung cấp sẵn model weights (`.keras`) và **TFLite** (`.tflite`)

---

## 🧠 Kiến trúc mô hình

### Fast-SCNN ⭐ (Mô hình đề xuất)

```
Input (256×256×3)
  → Learning to Downsample (Conv + DSConv, stride 2)
  → Global Feature Extractor (Bottleneck blocks + Pyramid Pooling)
  → Feature Fusion Module
  → Classifier (DSConv + Conv 1×1 sigmoid)
  → Bilinear Upsample → Output (256×256×1)
```

### ENet (Mô hình so sánh)

```
Input (256×256×3)
  → Initial Block (Conv + MaxPool concat)
  → Encoder (Bottleneck blocks, downsample ×2)
  → Decoder (Transpose Conv, upsample ×3)
  → Conv2DTranspose sigmoid → Output (256×256×1)
```

---

## 📁 Cấu trúc dự án

```
.
├── models/                           # Kiến trúc mô hình + weights
│   ├── fast_scnn.py                  # Định nghĩa kiến trúc Fast-SCNN
│   ├── enet.py                       # Định nghĩa kiến trúc ENet
│   ├── final_model_fast.keras        # Pretrained Fast-SCNN (~2.5 MB)
│   ├── final_model_enet.keras        # Pretrained ENet (~0.8 MB)
│   ├── fast_scnn_model.tflite        # TFLite INT8 Fast-SCNN (~636 KB)
│   └── enet_model.tflite             # TFLite INT8 ENet (~204 KB)
├── splits/
│   ├── train.txt                     # Danh sách file train
│   ├── val.txt                       # Danh sách file validation
│   └── test.txt                      # Danh sách file test
├── prepare_plantseg_binary.py        # Tiền xử lý PlantSeg → binary mask
├── tf_dataset.py                     # TF data pipeline (load, augment, batch)
├── check_dataset.py                  # Kiểm tra tính toàn vẹn dataset
├── train.py                          # Script huấn luyện mô hình
├── evaluate.py                       # Đánh giá mô hình trên tập test
├── infer_image.py                    # Inference trên 1 ảnh
├── infer_folder.py                   # Inference batch trên cả thư mục
├── paper.tex                         # Mã nguồn Báo cáo nghiên cứu khoa học (LaTeX)
├── references.bib                    # Tài liệu tham khảo cho báo cáo (BibTeX)
├── figures/                          # Hình ảnh minh họa cho báo cáo
├── .gitignore
└── README.md
```

---

## ⚙️ Yêu cầu môi trường

| Thành phần | Phiên bản khuyến nghị |
|-----------|----------------------|
| Python | 3.9 – 3.11 |
| TensorFlow | ≥ 2.12 |
| NumPy | ≥ 1.23 |
| Pillow | ≥ 9.0 |
| pandas | ≥ 1.5 *(chỉ cần cho `prepare_plantseg_binary.py`)* |

---

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/ddinhvinh/DeepLearning_Raspberry.git
cd DeepLearning_Raspberry
```

### 2. Tạo virtual environment (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS / Raspberry Pi
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install tensorflow numpy Pillow pandas
```

> **Trên Raspberry Pi**, cài TensorFlow Lite runtime thay vì full TensorFlow:
>
> ```bash
> pip install tflite-runtime numpy Pillow
> ```

---

## 📦 Dataset

Dự án sử dụng **PlantSeg Dataset** — *A Large-Scale In-the-wild Dataset for Plant Disease Segmentation* (~1.1 GB).

### 🔗 Link tải

| Nguồn | Link |
|-------|------|
| **Zenodo** | [📥 Tải PlantSeg Dataset](https://zenodo.org/records/17719108) |

> **Trích dẫn:** Wei, Tianqi; Chen, Zhi; Yu, Xin; Chapman, Scott; Melloy, Paul; Huang, Zi. *"A Large-Scale In-the-wild Dataset for Plant Disease Segmentation"*. Zenodo, 2025. DOI: [10.5281/zenodo.17719108](https://doi.org/10.5281/zenodo.17719108)

### Cấu trúc dataset sau khi tải

Giải nén và đặt vào thư mục `raw/`:

```
raw/
└── plantseg_raw/
    ├── images/
    ├── annotations/
    └── Metadata.csv
```

### Tiền xử lý dataset

```bash
python prepare_plantseg_binary.py
```

Script này sẽ:
- Đọc `Metadata.csv` để xác định split (train / val / test)
- Chuyển annotation thành **binary mask** (0 / 255)
- Lưu vào `processed_binary/{train,val,test}/{images,masks}/`
- Tạo file danh sách `splits/{train,val,test}.txt`

### Kiểm tra dataset

```bash
python check_dataset.py
```

---

## 📘 Hướng dẫn sử dụng

### 🏋️ Huấn luyện (Training)

```bash
# Huấn luyện Fast-SCNN (mặc định — mô hình đề xuất)
python train.py --data_root processed_binary --model fast_scnn --epochs 50 --batch_size 8 --lr 0.001

# Huấn luyện ENet (để so sánh)
python train.py --data_root processed_binary --model enet --epochs 50 --batch_size 8 --lr 0.001
```

**Tham số chính:**

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--data_root` | `D:\dataset\processed_binary` | Đường dẫn dataset đã xử lý |
| `--model` | `fast_scnn` | Mô hình: `fast_scnn` hoặc `enet` |
| `--img_size` | `256` | Kích thước ảnh đầu vào |
| `--batch_size` | `8` | Batch size |
| `--epochs` | `50` | Số epoch |
| `--lr` | `0.001` | Learning rate |
| `--save_root` | `outputs` | Thư mục lưu kết quả |

**Kết quả huấn luyện** được lưu tại:

```
outputs/{model_name}/
├── checkpoints/best.keras     # Model tốt nhất (theo val IoU)
├── logs/train_log.csv         # Log huấn luyện
├── logs/                      # TensorBoard logs
└── saved_model/
    ├── final_model.keras      # Model cuối cùng
    └── saved_model/           # TF SavedModel format
```

---

### 📊 Đánh giá (Evaluate)

```bash
python evaluate.py \
    --data_root processed_binary \
    --model_path models/final_model_fast.keras \
    --img_size 256 \
    --batch_size 4
```

Kết quả hiển thị: **Loss**, **Binary Accuracy**, **Dice Coefficient**, **IoU Score**.

---

### 🔍 Inference đơn ảnh

```bash
python infer_image.py \
    --model_path models/final_model_fast.keras \
    --image_path path/to/your/image.jpg \
    --mask_path path/to/ground_truth.png \
    --output_dir outputs/infer
```

Output gồm: `*_image.png`, `*_pred_mask.png`, `*_overlay.png`, `*_gt_mask.png` (nếu có).

---

### 📂 Inference batch (cả thư mục)

```bash
python infer_folder.py \
    --model_path models/final_model_fast.keras \
    --image_dir processed_binary/test/images \
    --mask_dir processed_binary/test/masks \
    --limit 20 \
    --output_dir outputs/infer_batch
```

---

## 📈 Kết quả

Kết quả đánh giá trên tập test, sử dụng **mô hình TFLite INT8** chạy trực tiếp trên **Raspberry Pi 4**:

| Mô hình | Binary Acc | Dice | IoU | Params (.keras) | Kích thước TFLite | FPS (RPi4) |
|---------|-----------|------|-----|-----------------|-------------------|------------|
| **Fast-SCNN** ⭐ | **86.32%** | **0.5924** | **0.4710** | 2.5 MB (~1.11M) | 636 KB | **23.34** |
| ENet | 83.29% | 0.5473 | 0.4195 | 0.8 MB (~0.37M) | 204 KB | 7.94 |

> **Nhận xét:** Fast-SCNN vượt trội ENet ở tất cả các chỉ số đánh giá. Đặc biệt, tốc độ inference trên Raspberry Pi 4 của Fast-SCNN (**23.34 FPS**) nhanh gấp **~3 lần** so với ENet (7.94 FPS), cho thấy Fast-SCNN là lựa chọn phù hợp nhất cho ứng dụng real-time trên thiết bị edge.

---

## 📜 Báo cáo Nghiên cứu Khoa học

Dự án này bao gồm mã nguồn báo cáo nghiên cứu khoa học hoàn chỉnh được soạn thảo bằng **LaTeX**. Báo cáo trình bày chi tiết toàn bộ quá trình nghiên cứu, từ cơ sở lý thuyết, kiến trúc mô hình (Fast-SCNN và ENet), kỹ thuật lượng tử hóa (INT8), cho đến đánh giá hiệu năng suy luận thực tế trên Raspberry Pi 4.

- **File chính:** `paper.tex`
- **Tài liệu tham khảo:** `references.bib`
- **Hình ảnh:** Thư mục `figures/`

**Biên dịch báo cáo (sử dụng pdflatex):**
```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```
Hoặc có thể upload trực tiếp thư mục này lên **Overleaf** để biên dịch.

---

## 👥 Tác giả

| MSSV | Họ và Tên | GitHub |
|------|-----------|--------|
| A50757 | **Phạm Thùy Dương** | [@pduonng29](https://github.com/pduonng29) |
| A51067 | **Đỗ Đình Vinh** | [@ddinhvinh](https://github.com/ddinhvinh) |
| A49612 | **Hà Phạm Mai Linh** | [@hpmlinh26](https://github.com/hpmlinh26) |

- **Lĩnh vực:** Khoa học Kỹ thuật và Công nghệ
- **GVHD:** ThS. Ngô Mạnh Cường
- **Đơn vị:** Khoa Công nghệ Thông tin — Trường Đại học Thăng Long

---

## 📄 License

Dự án này được phát hành dưới [MIT License](LICENSE).
