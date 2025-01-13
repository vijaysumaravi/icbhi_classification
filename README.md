# Respiratory sounds classification using breathing audio
Summary: Summary: Using Time-dilated Convolutional Neural Networks (TDNNs) for breathing sound classification. Two setups evaluated: 2 class classification and 4 class classification. Mel-frequency filter banks are used as features. ECAPA-TDNN is used as the model. Model and Data-preprocessing are implemented in Speechbrain.


## Dataset: 
Dataset can be downloaded from ICBHI Challenge website(https://bhichallenge.med.auth.gr/). After downloading data, separate the audio and annotations into two different folders - `audio_data` and `annotation_data`. Rename labels file to `icbi_labels.txt` and data split file to `icbi_train_test_split.txt`. 

### Dataset Description and Analysis
Preliminary datasets analysis is done using the `data_analysis.py` script to understand data distribution across different meta-data variables. Pass the path to main dataset folder to the script to get the analysis results. Below are some tables describing how the data is distrbuted. 

#### Audio Data Analysis

Number of audio files: 920

Sampling Rate Distribution
| Sampling Rate (Hz) | Count | Percentage |
|-------------------|-------|------------|
| 44100             | 824   | 89.6%      |
| 4000              | 90    | 9.8%       |
| 10000             | 6     | 0.7%       |


Chest Location Distribution
| Location | Count | Percentage |
|----------|-------|------------|
| Ar       | 168   | 18.3%      |
| Al       | 162   | 17.6%      |
| Pl       | 139   | 15.1%      |
| Pr       | 132   | 14.3%      |
| Tc       | 130   | 14.1%      |
| Lr       | 112   | 12.2%      |
| Ll       | 77    | 8.4%       |


Equipment Distribution
| Equipment      | Count | Percentage |
|---------------|-------|------------|
| AKGC417L.wav  | 646   | 70.2%      |
| Meditron.wav  | 127   | 13.8%      |
| LittC2SE.wav  | 87    | 9.5%       |
| Litt3200.wav  | 60    | 6.5%       |

Channel Distribution
| Channels | Count | Percentage |
|----------|-------|------------|
| 1        | 920   | 100.0%     |

Duration Statistics
| Statistic | Value (seconds) |
|-----------|----------------|
| Minimum   | 7.86           |
| Maximum   | 86.20          |
| Mean      | 21.49          |
| Median    | 20.00          |



#### Disease Analysis

Number of classes: 8

Class Distribution
| Class Label    | Count |
|---------------|-------|
| COPD          | 64    |
| Healthy       | 26    |
| URTI          | 14    |
| Bronchiectasis| 7     |
| Pneumonia     | 6     |
| Bronchiolitis | 6     |
| LRTI          | 2     |
| Asthma        | 1     |

Total number of patients: 126


#### Official Data Split Analysis
Number of train utterances: 539
Number of test utterances: 381

Speaker Distribution
| Split          | Unique Speakers |
|----------------|----------------|
| Train          | 79             |
| Test           | 49             |
| Total Unique*  | 126            |
*Note: Two speakers ('156', '218') appear in both train and test sets

#### Meta Analysis

Respiratory Cycle Statistics
| Statistic            | Value     |
|---------------------|-----------|
| Total number of cycles | 6,898     |
| Total duration (s)  | 18,628.112|
| Minimum duration (s)| 0.200     |
| Maximum duration (s)| 16.163    |
| Median duration (s) | 2.537     |


Respiratory Events Count and Duration
| Event Type              | Count | Percentage | Duration (s) | Duration % |
|------------------------|-------|------------|--------------|------------|
| Cycles with only crackles | 1,864 | 27.0%     | 5,190.4     | 27.9%     |
| Cycles with only wheezes  | 886   | 12.8%     | 2,394.8     | 12.9%     |
| Cycles with both         | 506   | 7.3%      | 1,548.3     | 8.3%      |
| Cycles with either       | 3,256 | 47.2%     | 9,133.5     | 49.0%     |
| Cycles with neither      | 3,642 | 52.8%     | 9,494.6     | 51.0%     |


Breathing Cycle Distribution by Disease Label
| Disease Label    | Normal | Abnormal | Total |
|-----------------|--------|-----------|--------|
| URTI            | 214    | 29        | 243    |
| Healthy         | 303    | 19        | 322    |
| Asthma          | 2      | 4         | 6      |
| COPD            | 2725   | 3021      | 5746   |
| LRTI            | 31     | 1         | 32     |
| Bronchiectasis  | 51     | 53        | 104    |
| Pneumonia       | 240    | 45        | 285    |
| Bronchiolitis   | 76     | 84        | 160    |
| Total           | 3642   | 3256      | 6898   |



## Data Preprocessing and Feature Extraction
- Removed files which have sampling rate other than 44100
- Chunked the segments with win_len and hop parameters as audio_processing_options in train.yaml
- Split the data into train, validation and test sets with the split_type and split_variable as train.yaml. Each split is stratified using label( abnormal and normal) as stratification variable.
- split_utterance: split all the segments randomly into train, validation and test sets (0.7, 0.1, 0.2)
- split_patient: split the patients into train, validation and test sets (0.7, 0.1, 0.2) and then take all the segments from the patients in the train, validation and test sets. Patients don't overlap. 

Dataset Statistics for utterance split
| Metric              | Combined | Train  | Valid  | Test   |
|--------------------|----------|--------|--------|--------|
| # Patients         | 109      | 109    | 102    | 109    |
| # Normal/Abnormal  | 6596/6666| 4617/4665| 660/667| 1319/1334|
| # Total Utterances | 13262    | 9282   | 1327   | 2653   |
| Duration (hours)   | 7.12     | 4.98   | 0.71   | 1.43   |
| Mean/Median Dur (s)| 1.9/2.0  | 1.9/2.0| 1.9/2.0| 1.9/2.0|


Dataset Statistics for patient split
| Metric              | Combined | Train  | Valid  | Test   |
|--------------------|----------|--------|--------|--------|
| # Patients         | 109      | 65     | 22     | 22     |
| # Normal/Abnormal  | 2956/2865| 1823/1319| 757/703| 376/843|
| # Total Utterances | 5821     | 3142   | 1460   | 1219   |
| Duration (hours)   | 4.30     | 2.43   | 0.99   | 0.88   |
| Mean/Median Dur (s)| 2.7/2.5  | 2.8/2.6| 2.4/2.4| 2.6/2.5|

- Feature Extraction: We utilize Mel-frequency cepstral coefficients (MFCCs) as the primary feature for breathing sound classification. The feature extraction process involves computing a set of 128 Mel-frequency bands (n_mels) from the audio signals, using a Fast Fourier Transform (FFT) size of 2048 (n_fft). The analysis window length is set to 46 samples (win_length), with a hop length of 12 samples (hop_length) to ensure overlap and smooth transitions between frames. Additionally, the feature set includes delta and double-delta coefficients (deltas: True), which capture the temporal dynamics of the audio signals. This comprehensive feature set is designed to effectively represent the acoustic characteristics of normal and abnormal breathing sounds for classification tasks.

- Data Augmentation: Signal-level data augmentation is applied to the training set using SpeechBrain's Augmenter module. Each audio sample has a chance to receive between 1 to 3 augmentations, applied in random order. The augmentation pipeline includes speed perturbation (modifying playback speed), frequency dropping (randomly masking frequency bands), and time masking (dropping random chunks of the signal). The augmented samples are concatenated with the original samples to maintain the original data distribution while expanding the dataset. This comprehensive augmentation strategy helps improve model robustness and generalization by introducing controlled variations in the training data.

## Model

The breathing sound classification model is built using the ECAPA-TDNN architecture, which is well-suited for processing sequential audio data. The model's embedding layer is configured with an input size of 384, calculated as the product of 128 Mel-frequency bands (`n_mels`) and 3, accounting for the inclusion of delta and double-delta features. The architecture consists of three convolutional layers with channel sizes of 256, 256, and 512, and kernel sizes of 5, 3, and 1, respectively. These layers are designed with dilations of 1, 2, and 1 to capture temporal patterns at different scales. An attention mechanism with 32 channels is employed to focus on the most relevant features, followed by a linear layer with 64 neurons. A dropout rate of 50% is applied to prevent overfitting. The classifier component of the model takes the 64-dimensional embeddings and outputs predictions for the two classes (normal and abnormal breathing sounds), leveraging the robust feature representations learned by the ECAPA-TDNN.

## Training
The training setup for the breathing sound classification task is meticulously configured to optimize the performance of the ECAPA-TDNN model. The training process is set to run for 100 epochs, with a batch size of 32 to balance computational efficiency and model convergence. The learning rate is initialized at 0.0001, with a weight decay of 0.0002 to regularize the model and prevent overfitting. A cyclic learning rate scheduler is employed, operating in a `triangular2` mode with a base learning rate of 0.00001 and a maximum learning rate of 0.0001, adjusting the learning rate dynamically to enhance training stability. The scheduler's step size is set to 458 (based on the number of samples in the dataset - 13262), and a gamma value of 0.9998 is used to gradually reduce the learning rate over time.

The data is shuffled before each epoch to ensure diverse mini-batches, and the sample rate for audio processing is set at 44,100 Hz. The model is trained using the Adam optimizer, which is well-suited for handling sparse gradients and non-stationary objectives. Checkpoints are saved every 15 minutes to safeguard against data loss and facilitate model recovery. The training process is logged to track progress and performance metrics, providing insights into the model's learning dynamics. This comprehensive training setup is designed to effectively harness the capabilities of the ECAPA-TDNN model for classifying breathing sounds into normal and abnormal categories.

The training progress is monitored and logged for each epoch, tracking multiple key metrics. Each log entry includes the epoch number, current learning rate, training loss, validation loss, and validation error rate. The format follows: `Epoch: {n}, lr: {learning_rate} - train loss: {train_loss} - valid loss: {valid_loss}, valid error: {valid_error}`. Training logs are saved to `train_log.txt` in the experiment's results directory. At the end of training, the best performing model using validation error as the metric is automatically loaded and evaluated on the test set, with results appended to the log file and stored in `eval_metrics.txt`.

Training model for 100 epochs takes around 1 hours on a single A6000GPU. Training Logs are stored in `log.txt` and `train_log.txt` in the experiment's results directory.

## Evaluation Metrics
The evaluation of the breathing sound classification model is comprehensively documented through both quantitative metrics and visual plots. The `compute_eval_metrics` function calculates and records key performance metrics such as the F1-macro score, sensitivity, specificity, and an average score, along with the confusion matrix components (True Negatives, False Positives, False Negatives, and True Positives). These metrics are crucial for understanding the model's precision, recall, and overall discriminative power, and are saved in an `eval_metrics.txt` file within the specified output folder for easy access and analysis.

In addition to these metrics, the training process is visually represented through loss curves, which are plotted and saved as `training_curves.png`. These plots display the moving averages of training and validation losses, as well as validation error over the epochs, providing a clear view of the model's learning dynamics. The loss curves help in diagnosing issues such as overfitting or underfitting by showing how the model's performance evolves over time. Together, these metrics and plots offer a comprehensive evaluation framework, ensuring that the model is both quantitatively and qualitatively assessed for its ability to classify normal and abnormal breathing sounds effectively.

## Results

| Task | Model | F1-macro | Sensitivity | Specificity | Score | TN | FP | FN | TP |
|------|-------|----------|-------------|-------------|-------|----|----|----|----|
| Breathing (Abnormal vs Normal) | baseline-utterance (2.5s_75pc) | 0.763 | 0.640 | 0.894 | 0.767 | 1179 | 140 | 480 | 854 |
| Breathing (Abnormal vs Normal) | baseline-patient (7s_50pc) | 0.567 | 0.453 | 0.827 | 0.640 | 311 | 65 | 461 | 382 |
| Breathing (Abnormal vs Normal) | baseline-patient (7s_50pc) + data-aug (3x) | 0.540 | 0.394 | 0.867 | 0.630 | 326 | 50 | 511 | 332 |
| Breathing (Crackle, Wheeze, Both, Normal) | baseline-patient (7s_50pc) | 0.2827 | 0.3206 | 0.8094 | 0.565 | - | - | - | - |

Detailed Results for 4-class classification:

Per-class Metrics:
| Class | Sensitivity | Specificity | Average |
|-------|------------|-------------|---------|
| Normal | 0.5665 | 0.7264 | 0.6465 |
| Crackles | 0.6991 | 0.5290 | 0.6141 |
| Wheezes | 0.0167 | 0.9824 | 0.4996 |
| Both | 0.0000 | 1.0000 | 0.5000 |
| **Mean** | **0.3206** | **0.8094** | **0.5650** |

Confusion Matrix:
```
[[298 223   5   0]
 [ 65 151   0   0]
 [ 21  38   1   0]
 [  4  40   9   0]]
```
These results are in line with the results reported in the literature [1].

Training Loss Plots - 2-class classification, utterance split:
![Training curves](ICBHI/breathing_classification_2_classes/results_utterance_split/ECAPA-TDNN/42/training_curves.png)

Training Loss Plots - 2-class classification, patient split:
![Training curves](ICBHI/breathing_classification_2_classes/results_patient_split/ECAPA-TDNN/42/training_curves.png)



## Discussion
The results show that the ECAPA-TDNN model is able to classify breathing sounds into normal and abnormal categories with decent accuracy. When split across utterances, the model is able to achieve a sensitivity of 0.640 and specificity of 0.894 for the 2-class classification task. When split across patients, it achieves a sensitivity of 0.3206 and specificity of 0.8094. This indicates that in the first case, the model may be overfitting to the channel data which is not representative of the overall data. In the second case, the model is able to generalize better to the overall data. When data augmentation is applied, the model is able to achieve a sensitivity of 0.540 and specificity of 0.867 for the 2-class classification task. Unfortunately the model is achieve better performance with data augmentation. From analyzing the loss curves during training, it is possible that the model is overfitting to the majority class when data augmentation is applied since data augmentation is applied to both classes. 

As expected, for the 4-class classification task, the model performance is not as good as the 2-class classification task. The model is able to achieve a sensitivity of 0.5665 and specificity of 0.7264. The class with the least number of samples, 'both' i.e. both crackles and wheezes, is the most difficult to classify. As none of the samples are classified as 'both', the sensitivity for this class is 0. The 'normal' and 'crackle' class have better performance. 

Overfitting was the most common problem in this project. The model was able to achieve good performance on the training set, but the performance on the validation and test sets were not as good. Several attempts were made to improve the model performance, including data augmentation, changing the model architecture by reducing number of parameters, and tuning the hyperparameters such as learning rate, weight decay, batch size, and step size. Other possible solutions which could improve the model performance but were not explored in this project are: loss modification (focal loss), or dedicated data augmentation for minority classes.

## Inference (steps to reproduce the results)
1. Install Speechbrain and matplotlib (pip install speechbrain matplotlib). The environments.yaml file is provided in the repo. 
2. Clone this repo
3. move the ICBHI folder along with its contents to the Speechbrain 'recipes' folder
3. cd to recipes/ICBHI/breathing_classification_2_classes/
4. Download the dataset from https://bhichallenge.med.auth.gr/ and separate the audio and annotations into two different folders - `audio_data` and `annotation_data`. Rename labels file to `icbi_labels.txt` and data split file to `icbi_train_test_split.txt`. (Dataset is already provided in the repo)
5. In the hparams/train_utterance.yaml and hparams/train_patients.yaml, change the `data_folder` to the path where you have stored the audio and annotation data.
6. The model checkpoints are stored in the respective task folders. For examples, the model for 2-class patient split evaluation is stored in `breathing_classification_2_classes/results_patient_split/ECAPA-TDNN/42/save`.
7. Run the inference script: `python train.py hparams/train_utterance.yaml` for utterance split evaluation. Since the training is completed, the model is loaded from the checkpoint and evaluated on the test set.
8.  Run the inference script: `python train.py hparams/train_patients.yaml` for patient split evaluation
9. Results are stored in `results_patient_split/ECAPA-TDNN/42/eval_metrics.txt` and `results_utterance_split/ECAPA-TDNN/42/eval_metrics.txt`
10. For 4-class classification, run the inference script: `python train.py hparams/train_4_classes.yaml` in the breathing_classification_4_classes folder.

## References
[1] Gairola, S., Tom, F., Kwatra, N., & Jain, M. (2021, November). Respirenet: A deep neural network for accurately detecting abnormal lung sounds in limited data setting. In 2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) (pp. 527-530). IEEE.
