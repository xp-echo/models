# Overview
This is the repository for the multi-label classification tasks.

# Directory tree
Set directories as follows.

Nervus (this repository)
  └materials (make this directory when you use)  
    └images (this directory has image files for CNN.)  
    |  └png_128  
    |    └AAA.png  
    |    └BBB.png  
    └splits  
      └trial.csv  

- CSV (in this case trial.csv) must contain columns named `id_XXX`, `filepath`, `label_XXX`, and `split`. Detailed explanation is shown below.

# Brief Usage
## Training phase
```python train.py --task classification --csv_name trial.csv --model B6 --criterion CEL --optimizer RAdam --epochs 100 --batch_size 256 --augmentation trivialaugwide --input_channel 1 --save_weight best --gpu_ids 0,1,2```

- `task` determines your task. You can use classification or regression. When classification, you should use CEL as the `criterion` while, when regression, you should use MSE/rMSE/MAE as the `criterion`.
- `csv_name` must be the csv name in the materials/splits directory. Details about this csv are shown in another section.
- `model` used to set the architecture of the model. B0 means EfficientNetB0, while B6 means EfficientNetB6. You can choose many models other than these. 
- `epochs` determines epochs for training.
- `batch_size` determines batch sizes for training.
- `augmentation` determines augmentation for training.
- `input_channel` determines number of channels for the image. 1 means gray scale image while 3 means rgb image.
- `gpu_ids` determines how many GPUs you use. Default is "-1" which means CPU only.
- `save_weight` decides how often you save weights. Default is "best" which means the weight when the loss value is the lowest is saved. If `save_weight` is specified as "each", weights are saved each time the total loss decreases.

## Testing phase
```python test.py --test_datetime yyyy-mm-dd-HH-MM-SS```

The directory of yyyy-mm-dd-HH-MM-SS will be made after training.


# Detailed Preparation
## CSV
This is the csv which we show as trial.csv in the brief usage section.
CSV must contain columns named `id_XXX`, `Institution`, `ExamID`, `filepath`, `label_XXX`, and `split`.

Examples:
| id_uniq | Institution    | ExamID | filepath        | label_mr   | label_as  | split  |
| -----   | -------------- | ------ | -----------     | ---------  | --------- |--------|
| 0001    | Institution_A  | 0001   | png_128/AAA.png | positive   | positive  |  train |
| 0002    | Institution_A  | 0002   | png_128/BBB.png | negative   | negative  |  val   |
| 0003    | Institution_A  | 0003   | png_128/CCC.png | positive   | positive  |  train |
| 0004    | Institution_B  | 0001   | png_128/DDD.png | positive   | negative  |  test  |
| 0005    | Institution_B  | 0002   | png_128/EEE.png | negative   | positive  |  train |
| 0006    | Institution_B  | 0003   | png_128/FFF.png | positive   | negative  |  train |
| 0007    | Institution_B  | 0004   | png_128/GGG.png | negative   | positive  |  train |
| 0008    | Institution_C  | 0001   | png_128/HHH.png | negative   | negative  |  val   |
| 0009    | Institution_C  | 0002   | png_128/III.png | positive   | negative  |  test  |
| :       | :              | :      | :               | :          | :         | :      |

Note:
- `id_XXX` must be unique throughout the csv.
- `Institution` should be an institution name.
- `ExamID` should be unique for each institution.
- `filepath` is a path to images for the model.
- `label_XXX` should have a classification target. Any name is available. If you use more than two `label_XXX` columns, it will be automatically recognize multi-label classification and automatically prepare the proper number of classifiers (FCs).
- `split` should have `train`, `val`, and `test`.


# CUDA VERSION
CUDA Version = 11.3, 11.4

# Citation
The manuscript containing this repository is been submitting.
After the peer-reviwing process, we will cite the paper here.
