# 🌿 Plant Segmentation on Raspberry Pi 4

> Nghiên cứu triển khai mô hình **Semantic Segmentation** (Fast-SCNN & ENet) cho bài toán phân vùng bệnh cây trồng trên **Raspberry Pi 4**, sử dụng TensorFlow / TensorFlow Lite.

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

Dự án này tập trung vào việc huấn luyện và triển khai hai mô hình **lightweight semantic segmentation** cho bài toán **phân vùng nhị phân (binary segmentation)** bệnh cây trồng:

| Mô hình | Đặc điểm |
|---------|-----------|
| **Fast-SCNN** | Learning to Downsample + Feature Fusion + Pyramid Pooling |
| **ENet** | Encoder-Decoder hiệu quả với Bottleneck blocks + PReLU |

Cả hai mô hình đều được thiết kế nhẹ, phù hợp triển khai trên thiết bị **edge** như Raspberry Pi 4.

### Tính năng chính

- ✅ Huấn luyện với **BCE + Dice Loss** kết hợp
- ✅ Đánh giá bằng **Dice Coefficient** và **IoU Score**
- ✅ Data augmentation (flip, brightness)
- ✅ Hỗ trợ inference đơn ảnh hoặc cả thư mục
- ✅ Xuất overlay trực quan (ảnh gốc + mask dự đoán)
- ✅ Cung cấp sẵn model weights (`.keras`) và **TFLite** (`.tflite`) để chạy trực tiếp trên Raspberry Pi

---

## 🧠 Kiến trúc mô hình

### Fast-SCNN

```
Input (256×256×3)
  → Learning to Downsample (Conv + DSConv, stride 2)
  → Global Feature Extractor (Bottleneck blocks + Pyramid Pooling)
  → Feature Fusion Module
  → Classifier (DSConv + Conv 1×1 sigmoid)
  → Bilinear Upsample → Output (256×256×1)
```

### ENet

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
│   ├── final_model_fast.keras        # Pretrained Fast-SCNN (~6.7 MB)
│   ├── final_model_enet.keras        # Pretrained ENet (~2.4 MB)
│   ├── fast_scnn_model.tflite        # TFLite Fast-SCNN cho RPi (~0.6 MB)
│   └── enet_model.tflite             # TFLite ENet cho RPi (~0.2 MB)
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
# Huấn luyện Fast-SCNN (mặc định)
python train.py --data_root processed_binary --model fast_scnn --epochs 50 --batch_size 8 --lr 0.001

# Huấn luyện ENet
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

> 💡 *Bổ sung kết quả sau khi huấn luyện xong.*

| Mô hình | Binary Acc | Dice | IoU | Params | FPS (RPi4) |
|---------|-----------|------|-----|--------|------------|
| Fast-SCNN | — | — | — | — | — |
| ENet | — | — | — | — | — |

---

## 👤 Tác giả

**Đình Vinh** — [@ddinhvinh](https://github.com/ddinhvinh)

**Hà Phạm Mai Linh** — [@hpmlinh26](https://github.com/hpmlinh26)

**Phạm Dương** — [@pduonng29](https://github.com/pduonng29)





