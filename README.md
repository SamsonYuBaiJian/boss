# BOSS Baseline Models
This repository contains baseline models for the BOSS dataset.

Our paper can be found here: https://arxiv.org/abs/2206.10665.

# Running the Code
## Data Setup
The BOSS dataset consists of:
- Video frames
- Belief labels for each frame
- Multiple input modalities, namely object-context relation, object bounding box, human pose and human gaze.

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
└── label.obj
```

## Training
1. Set your batch_size, num_epoch, save_path and gpu_id in `train.py`.
2. Uncomment only your desired model_type.
3. Based on the modality types you are using, only uncomment the appropriate inp_dim.
4. Set the frame paths for both train and validation sets to `path/to/dataset_folder/Train/frame` and `path/to/dataset_folder/Val/frame` respectively.
5. Set the label path to `path/to/dataset_folder/label.obj`.
6. Set the paths for your desired input modalities for both the train and validations sets (else leave them as None), similar to Step 4.

## Testing
1. For a specific experiment type, uncomment the appropriate test_frame_ids, else leave it as None.
2. For the median experiment type, uncomment line 46 and comment out line 47. For sequences with length above the median, change the second value in the tuple in line 46 to True, else change it to False. If you do not want to run testing for the median experiment type, leave median as None (i.e. comment out line 46 and uncomment 47).
2. Set your save_path and gpu_id in `test.py`.
3. Uncomment only your desired model_type and the import line for that model_type.
4. Based on the modality types you are using, only uncomment the appropriate inp_dim.
5. Set the load_model_path to the path of your desired saved model.
6. Set the frame path to `path/to/dataset_folder/Test/frame`.
7. Set the label path to `path/to/dataset_folder/label.obj`.
8. Set the paths for your desired input modalities (else leave them as None), similar to Step 4.