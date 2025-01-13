'''
    This file is used to analyze the ICBHI dataset and generate the necessary plots and statistics.
    Usage: python data_analysis.py <path_to_dataset>
    Authors: 
        - Vijay Ravi, 2025
'''

import pandas as pd
import numpy as np
import os
import sys
import glob
import librosa
import warnings
warnings.filterwarnings("ignore")


def disease_analysis(labels_file):
    '''
        This function is used to analyze the distribution of class labelsin the dataset.
    '''
    # load labels
    labels = pd.read_csv(labels_file, delimiter='\t', header=None)
    labels.columns = ['patient_id', 'label']

    # get the unique labels
    unique_labels = labels['label'].unique()

    print(f"Number of classes: {len(unique_labels)}")
    
    # get the count of each label
    label_counts = labels['label'].value_counts()
    
    # Print header
    print("Class Distribution")
    print("-" * 40)
    print(f"{'Class Label':<30} {'Count':>10}")
    print("-" * 40)
    
    # Print each label and count
    for label, count in label_counts.items():
        print(f"{str(label):<30} {str(count):>10}")
    print("-" * 40)

    # total number of patients
    total_patients = labels['patient_id'].nunique()
    print(f"Total number of patients: {total_patients}")


def data_split_analysis(data_split_file):
    '''
        This function is used to analyze the distribution of speakers and utterances in the training and testing dataset
        provided with the dataset.
    '''
    # load data split
    data_split = pd.read_csv(data_split_file, delimiter='\t', header=None)
    data_split.columns = ['utterance_id', 'split']

    # number of train and test utterances
    train_utterances = data_split[data_split['split'] == 'train'].shape[0]
    test_utterances = data_split[data_split['split'] == 'test'].shape[0]
    print(f"Number of train utterances: {train_utterances}")
    print(f"Number of test utterances: {test_utterances}")

    # number of train and test patients
    data_split['speaker_id'] = data_split['utterance_id'].apply(lambda x: x.split('_')[0])
    
    # Count unique speakers in train and test
    train_speakers = data_split[data_split['split'] == 'train']['speaker_id'].nunique()
    test_speakers = data_split[data_split['split'] == 'test']['speaker_id'].nunique()
    
    print("\nSpeaker Distribution")
    print("-" * 40)
    print(f"{'Split':<20} {'Unique Speakers':>20}")
    print("-" * 40)
    print(f"{'Train':<20} {str(train_speakers):>20}")
    print(f"{'Test':<20} {str(test_speakers):>20}")
    print(f"{'Total':<20} {str(train_speakers + test_speakers):>20}")
    print("-" * 40)

    # Get sets of unique speaker IDs for train and test
    train_speaker_set = set(data_split[data_split['split'] == 'train']['speaker_id'])
    test_speaker_set = set(data_split[data_split['split'] == 'test']['speaker_id'])
    
    # check if there is any overlap between train and test speakers
    overlap = train_speaker_set & test_speaker_set
    if overlap:
        print(f"Warning: These speakers are present in both train and test: {overlap}")
    else:
        print("No overlap between train and test speakers.")


def audio_data_analysis(audio_dir):
    '''
        This function is used to analyze the distribution of audio data in the dataset.
    '''

    # list all the files in the annotations directory
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    print(f"Number of audio files: {len(audio_files)}")

    # 
    audio_df = pd.DataFrame(columns=[
        'file_name',
        'speaker_id',
        'recording_index',
        'chest_location',
        'acquisition_mode',
        'equipment', 
        'duration',
        'sampling_rate',
        'num_channels'
    ])

    for audio_file in audio_files:
        # file name
        file_name = os.path.basename(audio_file)

        # speaker id
        speaker_id = file_name.split('_')[0]

        # recording index
        recording_index = file_name.split('_')[1]

        # chest location
        chest_location = file_name.split('_')[2]

        # acquisition mode
        acquisition_mode = file_name.split('_')[3]  

        # equipment
        equipment = file_name.split('_')[4]

        # duration
        duration = librosa.get_duration(path=audio_file)    

        # sampling rate
        sampling_rate = librosa.get_samplerate(audio_file)

        # number of channels - Fix: load audio and get shape
        y, _ = librosa.load(audio_file, sr=None, mono=False)
        num_channels = 1 if len(y.shape) == 1 else y.shape[0]

        # append to the dataframe using concat instead of append
        new_row = pd.DataFrame({
            'file_name': [file_name],
            'speaker_id': [speaker_id],
            'recording_index': [recording_index],
            'chest_location': [chest_location],
            'acquisition_mode': [acquisition_mode],
            'equipment': [equipment],
            'duration': [duration],
            'sampling_rate': [sampling_rate],
            'num_channels': [num_channels]
        })
        audio_df = pd.concat([audio_df, new_row], ignore_index=True)
    

    # unique sampling rates and their counts
    sampling_rate_counts = audio_df['sampling_rate'].value_counts()

    # unique chest locations and their counts
    chest_location_counts = audio_df['chest_location'].value_counts()

    # unique equipment and their counts
    equipment_counts = audio_df['equipment'].value_counts()

    # unique channels and their counts
    channels_counts = audio_df['num_channels'].value_counts()

    # min and max duration
    min_duration = audio_df['duration'].min()
    max_duration = audio_df['duration'].max()

    # Print sampling rate distribution
    print("\nSampling Rate Distribution")
    print("-" * 50)
    print(f"{'Sampling Rate (Hz)':<25} {'Count':>15} {'Percentage':>10}")
    print("-" * 50)
    for rate, count in sampling_rate_counts.items():
        percentage = (count / len(audio_df)) * 100
        print(f"{rate:<25} {count:>15} {percentage:>9.1f}%")
    print("-" * 50)

    # Print chest location distribution
    print("\nChest Location Distribution")
    print("-" * 50)
    print(f"{'Location':<25} {'Count':>15} {'Percentage':>10}")
    print("-" * 50)
    for loc, count in chest_location_counts.items():
        percentage = (count / len(audio_df)) * 100
        print(f"{loc:<25} {count:>15} {percentage:>9.1f}%")
    print("-" * 50)

    # Print equipment distribution
    print("\nEquipment Distribution")
    print("-" * 50)
    print(f"{'Equipment':<25} {'Count':>15} {'Percentage':>10}")
    print("-" * 50)
    for eq, count in equipment_counts.items():
        percentage = (count / len(audio_df)) * 100
        print(f"{eq:<25} {count:>15} {percentage:>9.1f}%")
    print("-" * 50)

    # Print channel distribution
    print("\nChannel Distribution")
    print("-" * 50)
    print(f"{'Channels':<25} {'Count':>15} {'Percentage':>10}")
    print("-" * 50)
    for ch, count in channels_counts.items():
        percentage = (count / len(audio_df)) * 100
        print(f"{ch:<25} {count:>15} {percentage:>9.1f}%")
    print("-" * 50)

    # Print duration statistics
    print("\nDuration Statistics")
    print("-" * 50)
    print(f"{'Statistic':<25} {'Value (seconds)':>15}")
    print("-" * 50)
    print(f"{'Minimum':<25} {min_duration:>15.2f}")
    print(f"{'Maximum':<25} {max_duration:>15.2f}")
    print(f"{'Mean':<25} {audio_df['duration'].mean():>15.2f}")
    print(f"{'Median':<25} {audio_df['duration'].median():>15.2f}")
    print("-" * 50)


def annotations_data_analysis(annotations_dir):
    '''
        This function is used to analyze the distribution of audio events in the dataset 
        using the annotation files.
    '''
    # list all the files in the annotations directory
    annotations_files = glob.glob(os.path.join(annotations_dir, "*.txt"))
    print(f"Number of annotations files: {len(annotations_files)}")

    # Initialize variables to store statistics
    all_cycle_durations = []
    total_crackles = 0
    total_wheezes = 0
    total_cycles = 0

    # Initialize counters for events
    total_only_crackles = 0
    total_only_wheezes = 0
    total_both = 0
    total_either = 0
    total_neither = 0

    # Initialize duration counters
    duration_only_crackles = 0.0
    duration_only_wheezes = 0.0
    duration_both = 0.0
    duration_either = 0.0
    duration_neither = 0.0

    # Process each annotation file
    for ann_file in annotations_files:
        annotations = pd.read_csv(ann_file, delimiter='\t', header=None)
        annotations.columns = ['start', 'end', 'crackles', 'wheezes']

        # Calculate durations for each cycle
        annotations['duration'] = annotations['end'] - annotations['start']
        cycle_durations = annotations['duration']
        all_cycle_durations.extend(cycle_durations.tolist())

        # Count total cycles
        total_cycles += len(annotations)
        
        # Create boolean masks for each category
        only_crackles_mask = (annotations['crackles'] == 1) & (annotations['wheezes'] == 0)
        only_wheezes_mask = (annotations['crackles'] == 0) & (annotations['wheezes'] == 1)
        both_mask = (annotations['crackles'] == 1) & (annotations['wheezes'] == 1)
        either_mask = (annotations['crackles'] == 1) | (annotations['wheezes'] == 1)
        neither_mask = (annotations['crackles'] == 0) & (annotations['wheezes'] == 0)

        # Count events
        total_only_crackles += only_crackles_mask.sum()
        total_only_wheezes += only_wheezes_mask.sum()
        total_both += both_mask.sum()
        total_either += either_mask.sum()
        total_neither += neither_mask.sum()

        # Sum durations
        duration_only_crackles += annotations.loc[only_crackles_mask, 'duration'].sum()
        duration_only_wheezes += annotations.loc[only_wheezes_mask, 'duration'].sum()
        duration_both += annotations.loc[both_mask, 'duration'].sum()
        duration_either += annotations.loc[either_mask, 'duration'].sum()
        duration_neither += annotations.loc[neither_mask, 'duration'].sum()

    # Calculate statistics
    min_duration = min(all_cycle_durations)
    max_duration = max(all_cycle_durations)
    median_duration = np.median(all_cycle_durations)
    total_duration = sum(all_cycle_durations)

    # Print results
    print("\nRespiratory Cycle Statistics")
    print("-" * 50)
    print(f"{'Statistic':<30} {'Value':>20}")
    print("-" * 50)
    print(f"{'Total number of cycles':<30} {total_cycles:>20}")
    print(f"{'Total duration (s)':<30} {total_duration:>20.3f}")
    print(f"{'Minimum duration (s)':<30} {min_duration:>20.3f}")
    print(f"{'Maximum duration (s)':<30} {max_duration:>20.3f}")
    print(f"{'Median duration (s)':<30} {median_duration:>20.3f}")
    print("-" * 50)

    print("\nRespiratory Events Count and Duration")
    print("-" * 75)
    print(f"{'Event Type':<30} {'Count':>10} {'Percentage':>10} {'Duration (s)':>12} {'Dur %':>10}")
    print("-" * 75)
    
    total_duration = sum(all_cycle_durations)
    
    print(f"{'Cycles with only crackles':<30} {total_only_crackles:>10} "
          f"{(total_only_crackles/total_cycles)*100:>9.1f}% {duration_only_crackles:>12.1f} "
          f"{(duration_only_crackles/total_duration)*100:>9.1f}%")
    
    print(f"{'Cycles with only wheezes':<30} {total_only_wheezes:>10} "
          f"{(total_only_wheezes/total_cycles)*100:>9.1f}% {duration_only_wheezes:>12.1f} "
          f"{(duration_only_wheezes/total_duration)*100:>9.1f}%")
    
    print(f"{'Cycles with both':<30} {total_both:>10} "
          f"{(total_both/total_cycles)*100:>9.1f}% {duration_both:>12.1f} "
          f"{(duration_both/total_duration)*100:>9.1f}%")
    
    print(f"{'Cycles with either':<30} {total_either:>10} "
          f"{(total_either/total_cycles)*100:>9.1f}% {duration_either:>12.1f} "
          f"{(duration_either/total_duration)*100:>9.1f}%")
    
    print(f"{'Cycles with neither':<30} {total_neither:>10} "
          f"{(total_neither/total_cycles)*100:>9.1f}% {duration_neither:>12.1f} "
          f"{(duration_neither/total_duration)*100:>9.1f}%")
    
    print("-" * 75)


def meta_analysis(labels_file, annotations_dir):
    '''
        This function analyzes the relationship between disease labels and breathing cycles (normal vs abnormal).
    '''
    # Create speaker2label dictionary from labels file
    labels_df = pd.read_csv(labels_file, delimiter='\t', header=None)
    labels_df.columns = ['patient_id', 'label']
    speaker2label = dict(zip(labels_df['patient_id'], labels_df['label']))

    # Initialize dictionary to store counts for each label
    label_stats = {}
    for label in labels_df['label'].unique():
        label_stats[label] = {'normal': 0, 'abnormal': 0}

    # Process each annotation file
    annotation_files = glob.glob(os.path.join(annotations_dir, "*.txt"))
    for ann_file in annotation_files:
        # Get speaker ID from filename
        speaker_id = int(os.path.basename(ann_file).split('_')[0])
        if speaker_id not in speaker2label:
            print(f"Warning: Speaker {speaker_id} not found in labels")
            continue

        # Get label for this speaker
        label = speaker2label[speaker_id]

        # Read annotations
        annotations = pd.read_csv(ann_file, delimiter='\t', header=None)
        annotations.columns = ['start', 'end', 'crackles', 'wheezes']

        # Count normal and abnormal cycles
        abnormal_mask = (annotations['crackles'] == 1) | (annotations['wheezes'] == 1)
        normal_cycles = (~abnormal_mask).sum()
        abnormal_cycles = abnormal_mask.sum()

        # Add to statistics
        label_stats[label]['normal'] += normal_cycles
        label_stats[label]['abnormal'] += abnormal_cycles

    # Print results
    print("\nBreathing Cycle Distribution by Disease Label")
    print("-" * 55)
    print(f"{'Disease Label':<20} {'Normal':>10} {'Abnormal':>10} {'Total':>10}")
    print("-" * 55)

    total_normal = 0
    total_abnormal = 0
    
    for label, stats in label_stats.items():
        total = stats['normal'] + stats['abnormal']
        total_normal += stats['normal']
        total_abnormal += stats['abnormal']
        
        print(f"{str(label):<20} {stats['normal']:>10} {stats['abnormal']:>10} {total:>10}")
    
    # Print total row
    grand_total = total_normal + total_abnormal
    print("-" * 55)
    print(f"{'Total':<20} {total_normal:>10} {total_abnormal:>10} {grand_total:>10}")
    print("-" * 55)


if __name__ == "__main__":

    # Default data directory
    default_data_dir = "/u/home/s/spaceman/project-voicesafe/spapl/vijay/speechbrain/cough_audio/icbi/"
    
    # Use command line argument if provided, otherwise use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data_dir
    
    # path where audio files are stored - split the wav and txt files into audio and annotations. 
    audio_dir = os.path.join(data_dir, "audio_data")
    annotations_dir = os.path.join(data_dir, "annotation_data")
    labels_file = os.path.join(data_dir, "icbi_labels.txt")
    data_split_file = os.path.join(data_dir, "icbi_train_test_split.txt")

    # # disease analysis
    # disease_analysis(labels_file)

    # # data split analysis
    # data_split_analysis(data_split_file)

    # # audio data analysis
    # audio_data_analysis(audio_dir)

    # # annotations data analysis
    # annotations_data_analysis(annotations_dir)

    # # meta analysis
    # meta_analysis(labels_file, annotations_dir)







