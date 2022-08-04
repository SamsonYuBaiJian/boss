# BOSS Baseline Models
This repository contains baseline models for the BOSS dataset.

Our paper can be found here: https://arxiv.org/abs/2206.10665.

# Running the Code
## Data Setup
The BOSS dataset consists of video frames, belief labels for each frame, and multiple input modalities, namely object-context relation, object bounding box, human pose and human gaze.

The dataset folder has the following structure:
```
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
4. Change the frame paths for both train and validation sets to `dataset_folder/Train/frame` and `dataset_folder/Val/frame` respectively.
5. Change the label path to `dataset_folder/label.obj`.
6. Uncomment the paths for your desired input modalities, and set them for both the train and validations sets, similar to Step 4.

## Testing
1. For a specific experiment type, uncomment the appropriate test_frame_ids, else leave it as None.
2. For the median experiment type, uncomment line 46. For sequences with length above the median, change the second value in the tuple in line 46 to True, else change it to False. If you do not want to run testing for the median experiment type, leave median as None.
2. Set your save_path and gpu_id in `test.py`.
3. Uncomment only your desired model_type and the import line for that model_type.
4. Based on the modality types you are using, only uncomment the appropriate inp_dim.
5. Change the load_model_path to the path of your desired saved model.
6. Change the frame path to `dataset_folder/Test/frame`.
7. Change the label path to `dataset_folder/label.obj`.
8. Uncomment the paths for your desired input modalities, and set them for both the train and validations sets, similar to Step 4.