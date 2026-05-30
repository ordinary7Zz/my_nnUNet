# nnU-Net 2D 数据集与运行说明

这份说明记录 nnU-Net v2 的 2D 数据集格式，以及常用的数据准备、训练和推理命令。

## 1. nnU-Net 的 2D 数据集格式

nnU-Net v2 支持原生 2D 图像输入，不需要把 2D 图像强行转换成 3D。适合超声、X 光、显微图像、自然图像等二维分割任务。

### 数据集根目录

数据集必须放在 `nnUNet_raw` 下，目录命名格式如下：

```text
nnUNet_raw/
└── DatasetXXX_Name/
    ├── dataset.json
    ├── imagesTr/
    ├── labelsTr/
    └── imagesTs/   # 可选
```

- `DatasetXXX_Name`：`XXX` 是三位数字 ID，`Name` 是数据集名称
- `imagesTr`：训练图像
- `labelsTr`：训练标签
- `imagesTs`：测试图像，可选，nnU-Net 不用于训练
- `dataset.json`：描述通道和标签信息

### 文件命名规则

每个训练样本由一组图像和一张标签组成，文件名前缀必须一致：

```text
imagesTr/
├── case001_0000.png
├── case002_0000.png
└── case003_0000.png

labelsTr/
├── case001.png
├── case002.png
└── case003.png
```

规则：

- 图像文件名格式：`{CASE_IDENTIFIER}_{XXXX}.{FILE_ENDING}`
- 标签文件名格式：`{CASE_IDENTIFIER}.{FILE_ENDING}`
- `XXXX` 是通道编号，从 `0000` 开始
- 每个通道必须单独存一个文件
- 同一个 case 的所有输入图像必须尺寸一致、几何对齐
- 标签必须和图像同尺寸
- 背景类必须是 `0`
- 类别值必须连续，例如 `0, 1, 2, ...`

### 2D 超声图像的建议

如果你做甲状腺超声分割，通常可以按下面方式组织：

- 单通道灰度超声图：只用 `0000`
- 标签图：像素值为 `0` 表示背景，`1` 表示甲状腺或病灶，按你的任务继续扩展类别
- 图像和标签建议使用无损格式，例如 `.png`、`.tif`、`.nii.gz`
- 不要使用 `.jpg`，因为有压缩损失

### `dataset.json` 示例

```json
{
  "channel_names": {
    "0": "US"
  },
  "labels": {
    "background": 0,
    "thyroid": 1
  },
  "numTraining": 120,
  "file_ending": ".png"
}
```

如果你的数据是单通道超声图，`channel_names` 里通常写一个通道即可，比如 `US`、`gray` 或你习惯的名字。

---

## 2. 数据准备流程

### 第一步：整理数据

把原始图像和 mask 放到 nnU-Net 规定的目录中。

例如：

```text
nnUNet_raw/
└── Dataset101_ThyroidUS/
    ├── dataset.json
    ├── imagesTr/
    │   ├── us_001_0000.png
    │   ├── us_002_0000.png
    │   └── ...
    ├── labelsTr/
    │   ├── us_001.png
    │   ├── us_002.png
    │   └── ...
    └── imagesTs/
        ├── us_test_001_0000.png
        └── ...
```

### 第二步：检查完整性

第一次运行时建议加上 `--verify_dataset_integrity`：

```bash
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity
```

### 第三步：规划和预处理

推荐直接一条命令完成 fingerprint、planning 和 preprocessing：

```bash
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity
```

如果想拆开执行，也可以分别运行：

```bash
nnUNetv2_extract_fingerprint -d 101
nnUNetv2_plan_experiment -d 101
nnUNetv2_preprocess -d 101
```

预处理结果会写到：

```text
nnUNet_preprocessed/Dataset101_ThyroidUS/
```

---

## 3. 训练脚本使用

### Python 脚本方式

`my_scripts/train.py` 把原来的训练参数改成了脚本顶部变量。你只需要修改这些变量：

- `DATASET`
- `CONFIGURATION`
- `FOLD`
- `DATASET_JSON`
- `IMAGES_TR`
- `LABELS_TR`
- `IMAGES_TS`
- `DEVICE`

然后直接运行：

```bash
python my_scripts/train.py
```

如果你想改 trainer、plans、是否保存 `npz`、是否继续训练，也可以继续在脚本顶部改对应变量。

### 训练输出

训练结果一般在：

```text
nnUNet_results/Dataset101_ThyroidUS/<trainer>__<plans>__2d/fold_0
```

常见文件包括：

- `checkpoint_best.pth`
- `checkpoint_final.pth`
- `progress.png`
- `validation/summary.json`

---

## 4. 推理脚本使用

### Python 脚本方式

`my_scripts/infer.py` 同样把推理参数改成了脚本顶部变量。你只需要修改这些变量：

- `INPUT_FOLDER`
- `OUTPUT_FOLDER`
- `DATASET`
- `CONFIGURATION`
- `FOLDS`
- `DEVICE`

然后直接运行：

```bash
python my_scripts/infer.py
```

### 输入文件要求

推理输入文件命名方式必须和训练数据一致，例如：

```text
INPUT_FOLDER/
├── test_001_0000.png
├── test_002_0000.png
└── ...
```

注意：

- 文件后缀要和训练时一致
- 通道编号要和训练时一致
- 输入图像的读取格式必须和数据集一致

### 使用单个 fold 或 all fold

默认情况下，nnU-Net 会使用已训练的 5 个 fold 做集成。如果你想直接指定 `all` fold，可在 `my_scripts/infer.py` 里把 `FOLDS` 改成 `['all']`。

---

## 5. 常用工作流总结

### 新数据集标准流程

```bash
python my_scripts/train.py   # 修改脚本顶部的 DATASET / DATASET_JSON / IMAGES_TR / LABELS_TR / FOLD
python my_scripts/train.py   # 把 FOLD 改成 1、2、3、4 后重复
python my_scripts/infer.py   # 修改脚本顶部的 INPUT_FOLDER / OUTPUT_FOLDER / FOLDS
```

### 脚本入口

脚本入口如下，原生命令仍可用 `-h` 查看帮助：

```bash
python my_scripts/train.py
python my_scripts/infer.py
nnUNetv2_plan_and_preprocess -h
```

---

## 6. 适合甲状腺超声分割吗

适合。对于甲状腺超声图像分割，通常就是典型的 2D 语义分割任务，nnU-Net 的 2D 配置就是为这类数据准备的。

如果你后续需要，我可以继续帮你把你手上的超声数据整理成 nnU-Net 目录结构，或者直接生成一个可用的 `dataset.json` 模板。