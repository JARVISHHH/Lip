# Lip

## Project structure

The following is the directory tree for this project.

The datasets folder is not uploaded to GitHub.

```
lip/
├── code/
│   └── preprocess/
├── common/
│   └── predictors/
└── datasets/
    ├── alignments/
    │   ├── s1/
    │   ├── s2/
    │   ├── s3/
    │   └── ...
    ├── mouth/
    │   ├── train/
    │   │   ├── s3/
    │   │   ├── s4/
    │   │   └── ...
    │   └── val/
    │       ├── s1/
    │       ├── s2/
    │       └── ...
    └── videos/
        ├── s1/
        ├── s2/
        ├── s3/
        └── ...
```



## Commands

| command name      | usage                       | default |
| ----------------- | --------------------------- | ------- |
| --load-cache      | to load preprocessing cache | False   |
| --load-testcache  | to load test cache          | False   |
| --load-checkpoint | to load checkpoint          |         |
| --load-pretrain   | to load pretrain weights    |         |
| --train           | to train the model          | False   |
| --evaluate        | to test the model           | False   |



## Download datasets

https://drive.google.com/drive/folders/1nZ1Vg_fBOYAiu_cxt762WLC-A1YKBz5a?usp=drive_link

Download the datasets from the URL above to the right path according to the project structure. (videos are not uploaded since we will not be using it while training)



## How to modify and run

Run the following command line

```bash
git clone git@github.com:JARVISHHH/Lip.git
cd Lip/code
```

**We should now under the code folder.**

### Preprocess (for Haiyang and Yingtong)

To extract mouth images from videos, run the following command line

```bash
python extract_mouth_batch.py <video_path> <match_pattern> <target_path>
```

modify `<video_path>`, `<match_pattern>` and `<target_path>` to corresponding values.

For example, for the first speaker, we should run (we save `s1` to `val` according the LipNet paper)

```bash
python extract_mouth_batch.py ../datasets/videos/s1 *.mpg ../datasets/mouth/val/s1
```

and for the third speaker, we should run (we save `s3` to `train` according the LipNet paper)

```
python extract_mouth_batch.py ../datasets/videos/s3 *.mpg ../datasets/mouth/train/s3
```



### Model set up (for Jue)

Modify `architecture`, `metrics` in `model.py`, and customize the `callback` in `main.py`. The current `architecture` is the same with the model architecture in LipNet's GitHub https://github.com/rizkiarm/LipNet/blob/master/lipnet/model.py

We have several versions of the structures to be trained. See commands for comparasions.

Run

```bash
python main.py --load-testcache --train
```

to test if the code is runnable.

`--load-testcache` will load the preprocessing cache to quickly start training, but the test cache provided only contains a very small amount of data, it can only be used to test if the code is runnable.



### Train

Run

```
python main.py --train
```

It will first load all the data from datasets and then start to train the model.

See `main.py` for more arguments.

All hyperparameters are in `hyperparameters.py`.