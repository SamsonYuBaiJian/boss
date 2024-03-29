# BOSS Baseline Models
This repository contains baseline models for the BOSS dataset.

Website: https://sites.google.com/view/bossbelief/<br />
Dataset: https://drive.google.com/drive/folders/1b8FdpyoWx9gUps-BX6qbE9Kea3C2Uyua<br />
Paper: https://arxiv.org/abs/2206.10665

# Dataset Summary
The BOSS dataset consists of:
- Original and masked video frames
- Belief labels for each frame
- Multiple input modalities, namely object bounding box, object-context relation, human gaze and human pose

The dataset is split into three different sets: train, test and val.

The dataset folder has the following structure:
```
BOSS
├── Train
│   ├── bbox
│   ├── frame
│   ├── gaze
│   ├── masked
│   └── pose
├── Test
│   ├── bbox
│   ├── frame
│   ├── gaze
│   ├── masked
│   └── pose
├── Val
│   ├── bbox
│   ├── frame
│   ├── gaze
│   ├── masked
│   └── pose
├── README
├── OCRMap.txt
└── label
```

## Metadata
- The original frames are found in the frame folders and the masked frames are found in the masked folders.
- The belief labels are found in the label file.
- The object bounding boxes are found in the bbox folders.
- The object-context relations are found in OCRMap.txt.
- The human gaze data is found in the gaze folders.
- The human pose data is found in the pose folders.

More details can be found in Section 5.2 of our [paper](https://arxiv.org/abs/2206.10665).

# Running the Code
## Installing Dependencies
Run `pip install -r requirements.txt`.

## Training
1. Set your batch_size, num_epoch, save_path and gpu_id in `train.py`.
2. Uncomment only your desired model_type.
3. Based on the modality types you are using, only uncomment the appropriate inp_dim.
4. Set the frame paths for both train and validation sets to `path/to/dataset_folder/Train/frame` and `path/to/dataset_folder/Val/frame` respectively.
5. Set the label path to `path/to/dataset_folder/label`.
6. Set the paths for your desired input modalities for both the train and validations sets (else leave them as None), with paths similar to Step 4.
7. Run with `python train.py`.

## Testing
1. For a specific experiment type, uncomment the appropriate test_frame_ids, else leave it as None.
2. For the median experiment type, uncomment line 46 and comment out line 47. For sequences with length above the median, change the second value in the tuple in line 46 to True, else change it to False. If you do not want to run testing for the median experiment type, leave median as None (i.e. comment out line 46 and uncomment 47).
2. Set your save_path and gpu_id in `test.py`.
3. Uncomment only your desired model_type.
4. Based on the modality types you are using, only uncomment the appropriate inp_dim.
5. Set the load_model_path to the path of your desired saved model.
6. Set the frame path to `path/to/dataset_folder/Test/frame`.
7. Set the label path to `path/to/dataset_folder/label`.
8. Set the paths for your desired input modalities (else leave them as None), with paths similar to Step 4.
9. Run with `python test.py`.