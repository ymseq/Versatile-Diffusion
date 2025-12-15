# 数据准备指令与代码报告

## 数据准备指令
以下命令仅用于说明如何组织目录与拉取文件，不会在当前环境中自动下载；可根据网络环境替换下载源或使用断点续传工具。

### 1. 预创建目录
```bash
# 建议的统一数据根目录
mkdir -p /data/coco/images /data/coco/annotations
mkdir -p /data/coyo/images
mkdir -p /data/laion2b
mkdir -p /data/checkpoints/clip /data/checkpoints/lpips /data/fid_stats
```
- `/data/coco` 用于 COCO 图像与字幕标注，路径与 `configs/dataset/coco.yaml` 中的 `data_root`、`ann_path` 对应。
- `/data/coyo` 和 `/data/laion2b` 与 `coyo_subset`、`laion2b_subset` 配置保持一致，便于直接加载元数据文件。
- `/data/checkpoints`、`/data/fid_stats` 用于存放评估所需的外部资源（CLIP 模型、LPIPS 权重、FID 统计等）。

### 2. COCO 图像与标注
```bash
# 下载并解压图像（示例为 train2017/val2017，可按需精简）
wget http://images.cocodataset.org/zips/train2017.zip -O /data/coco/train2017.zip
unzip /data/coco/train2017.zip -d /data/coco/images

wget http://images.cocodataset.org/zips/val2017.zip -O /data/coco/val2017.zip
unzip /data/coco/val2017.zip -d /data/coco/images

# 下载字幕标注
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O /data/coco/ann.zip
unzip /data/coco/ann.zip -d /data/coco/annotations
```
- 下载后，`/data/coco/images/train2017`、`/data/coco/images/val2017` 应包含 `.jpg` 图片；字幕 JSON 需放在 `/data/coco/annotations`，以匹配 `configs/dataset/coco.yaml` 与评估配置 `configs/eval/i2t_sample.yaml`。
- 如只需子集，可仅解压需要的分割或自行裁剪元数据。

### 3. COYO 子集
```bash
# 假设已有 metadata.json（字段至少包含 image、caption）
# 将图片同步或下载到本地
rsync -av <存储端>/coyo/images/ /data/coyo/images/
# 如需通过 URL 下载，可按 metadata.json 的 image 字段循环 wget
```
- `meta_path` 在配置中指向 `/data/coyo/metadata.json`，图像路径可为绝对路径或相对 `data_root`（即 `/data/coyo/images`）。

### 4. LAION2B 子集
```bash
# 准备 metadata.tsv，至少包含 image\tcaption 两列
rsync -av <存储端>/laion2b/metadata.tsv /data/laion2b/metadata.tsv
# 可选：若已将图片缓存到本地，同步到 /data/laion2b/
rsync -av <存储端>/laion2b/images/ /data/laion2b/images/
```
- `metadata.tsv` 可逐行存储 `image\tcaption`，与 `lib/data_factory/factory.py` 的解析逻辑兼容；`data_root` 在配置中为 `/data/laion2b/`。

### 5. 评估依赖（FID/CLIP/LPIPS/参考字幕）
```bash
# FID 统计（示例：COCO val 统计）
wget <fid_stats_url> -O /data/fid_stats/coco_val.npz
# CLIP 权重（可使用 HF 镜像或官方权重）
wget <clip_weight_url> -O /data/checkpoints/clip/ViT-B-32.pt
# LPIPS 权重
wget <lpips_weight_url> -O /data/checkpoints/lpips/weights.pth
```
- `configs/eval/i2t_sample.yaml` 需要 `reference_path` 指向 COCO 验证字幕 JSON；若使用 FID/LPIPS，则需在相应评估配置中把路径改成下载位置。
- CLIP 模型名也可直接写框架支持的名称（如 `ViT-B-32`），但提前缓存权重可避免首轮加载时在线下载。

### 6. 配置文件路径检查
- 将 `configs/dataset/*.yaml` 与 `configs/eval/*.yaml` 中的 `data_root`、`ann_path/meta_path`、`reference_path`、`fid_stats_path`、`lpips_weight` 等字段替换为本地真实路径。
- 评估器在初始化时会检查文件存在性，缺失会直接抛出 `FileNotFoundError`，运行前务必确认文件到位。

### 7. 目录结构参考
```
/data
  ├─ coco
  │   ├─ images
  │   │   ├─ train2017/*.jpg
  │   │   └─ val2017/*.jpg
  │   ├─ annotations
  │   │   ├─ captions_train2017.json
  │   │   └─ captions_val2017.json
  │   └─ fid_stats/coco_val.npz  # 若需要 FID
  ├─ coyo
  │   ├─ images/...              # 与 metadata.json 对应
  │   └─ metadata.json
  ├─ laion2b
  │   ├─ metadata.tsv
  │   └─ images/...              # 可选，本地缓存
  └─ checkpoints
      ├─ clip/ViT-B-32.pt        # 或其他 CLIP 权重
      └─ lpips/weights.pth
```

## 代码报告
- **数据工厂**：`lib/data_factory/factory.py` 注册了数据集、变换、拼接与采样器，支持 JSON、逐行 JSON/TSV 的元数据读取，并提供默认训练/评估图像变换、分布式采样器和批处理逻辑。COCO、LAION 子集与图像变体数据集通过注册器暴露。 
- **评估器骨架**：`lib/evaluator/builder.py` 定义了通用 `BaseEvaluator`，负责聚合批次结果并保存均值；并注册了文本生成图像、图像变体和图像生成文本三类评估器，初始化时会校验 FID 统计、LPIPS 权重、参考字幕等路径是否存在。
- **配置示例**：`configs/dataset/` 与 `configs/eval/i2t_sample.yaml` 提供了 COCO/COYO/LAION 子集的默认路径和批处理设置，便于直接对照修改。
