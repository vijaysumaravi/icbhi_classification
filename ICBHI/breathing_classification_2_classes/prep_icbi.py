'''
    This file creates the data manifest files for the ICBI dataset.
    Authors: 
        - Vijay Ravi
    Date: 2025-01-11
'''

import json
import os
import random
import re
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedShuffleSplit
from pdb import set_trace as st
from collections import Counter
from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)
SAMPLERATE = 44100


def prep_icbi(audio_dir, 
              annotation_dir,
              save_json_train,
              save_json_valid,
              save_json_test, 
              audio_processing_options,
              split_type,
              split_variable,
              split_ratio,
              seed):
    '''
        This function prepares the ICBI dataset for training, validation and testing.
    '''
    # list of all audio files
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

    # files with sample rate not 44100 are removed.
    audio_files_valid = [f for f in audio_files if get_audio_sample_rate(f) == SAMPLERATE]
    logger.info(f"Number of valid audio files: {len(audio_files_valid)}")
    audio_files = audio_files_valid
    annotation_files = [os.path.join(annotation_dir, os.path.basename(f).replace('.wav', '.txt')) for f in audio_files]
    logger.info(f"Number of annotation files: {len(annotation_files)}")

    data_df = get_data_df(audio_files, annotation_files, audio_processing_options)
    
    train_df, validation_df, test_df = split_data(data_df, split_type, split_variable, split_ratio, seed)

    # save the dataframes to json files
    create_json(train_df, save_json_train)
    create_json(validation_df, save_json_valid)
    create_json(test_df, save_json_test)

    #print data statistics
    print_data_stats(data_df, train_df, validation_df, test_df)


def print_data_stats(data_df, train_df, validation_df, test_df):
    '''
    This function prints the data statistics in a table format showing:
    - Number of unique patients
    - Number of normal/abnormal utterances
    - Total utterances
    - Total duration (in hours)
    - Mean/median duration of segments (in seconds)
    For each split (combined, train, validation, test)
    '''
    # Create lists to store stats for each split
    splits = ['Combined', 'Train', 'Valid', 'Test']
    dfs = [data_df, train_df, validation_df, test_df]
    
    # Calculate stats for each split
    stats = []
    for df in dfs:
        # Basic stats
        n_patients = df['patient_id'].nunique()
        n_normal = len(df[df['utterance_label'] == 'normal'])
        n_abnormal = len(df[df['utterance_label'] == 'abnormal'])
        n_total = len(df)
        
        # Calculate durations
        segment_durations = [(end - start) for start, end in df['start_end_time']]
        total_duration = sum(segment_durations)
        duration_hours = total_duration / 3600  # Convert seconds to hours
        mean_duration = sum(segment_durations) / len(segment_durations)
        median_duration = sorted(segment_durations)[len(segment_durations)//2]
        
        stats.append([
            n_patients, 
            f"{n_normal}/{n_abnormal}", 
            n_total, 
            f"{duration_hours:.2f}",
            f"{mean_duration:.1f}/{median_duration:.1f}"
        ])
    
    # Print table header
    print("\nDataset Statistics:")
    print("-" * 85)
    print(f"{'Metric':<25} {'Combined':<15} {'Train':<15} {'Valid':<15} {'Test':<15}")
    print("-" * 85)
    
    # Print rows
    metrics = [
        "# Patients", 
        "# Normal/Abnormal", 
        "# Total Utterances", 
        "Duration (hours)",
        "Mean/Median Dur (s)"
    ]
    for i, metric in enumerate(metrics):
        row = [stat[i] for stat in stats]
        print(f"{metric:<25}", end="")
        for val in row:
            print(f"{str(val):<15}", end="")
        print()
    print("-" * 85)


def create_json(data_df, save_json_file):
    '''
        This function creates the json file for the ICBI dataset.
    '''
    json_dict = {}
    for idx, row in data_df.iterrows():
        json_dict[row['segment_id']] = {
            'wav': row['wav'],
            'start_end_time': row['start_end_time'],
            'utterance_label': row['utterance_label']
        }
    with open(save_json_file, 'w') as f:
        json.dump(json_dict, f, indent=4)
    logger.info(f"Saved {save_json_file} successfully!")


def split_data(data_df, split_type, split_variable, split_ratio, seed):
    if split_type == 'random':
        if split_variable == 'utterance':
            train_df, validation_df, test_df = split_random_utterance(data_df, split_ratio, seed)
        elif split_variable == 'patient':
            train_df, validation_df, test_df = split_random_patient(data_df, split_ratio, seed)
        else:
            raise ValueError(f"Invalid split variable: {split_variable}")
    elif split_type == 'official':
        pass
        # train_df, validation_df, test_df = split_official(data_df, split_ratio)
    else:
        raise ValueError(f"Invalid split type: {split_type}")
    
    return train_df, validation_df, test_df


def split_random_utterance(data_df, split_ratio, seed):
    """
    Split data into train, validation and test sets using random splitting of utterances in a stratified manner.
    patients may overlap between train, validation and test sets.
    Uses a two-step process:
    1. First split into (train+valid) and test
    2. Then split train+valid into train and validation
    
    Args:
        data_df: DataFrame containing the data
        split_ratio: tuple of (train, valid, test) ratios that sum to 1
    """
    logger.info(f"Splitting data into train, validation and test sets using random splitting of utterances in a stratified manner.")
    train_ratio, valid_ratio, test_ratio = split_ratio
    
    # First split: (train+valid) vs test
    first_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_valid_idx, test_idx = next(first_splitter.split(data_df, data_df['utterance_label']))
    
    train_valid_df = data_df.iloc[train_valid_idx]
    test_df = data_df.iloc[test_idx]
    
    # Second split: train vs valid
    # Adjust validation ratio to account for the reduced dataset size
    valid_ratio_adjusted = valid_ratio / (train_ratio + valid_ratio)
    second_splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio_adjusted, random_state=seed)
    train_idx, valid_idx = next(second_splitter.split(train_valid_df, train_valid_df['utterance_label']))
    
    train_df = train_valid_df.iloc[train_idx]
    validation_df = train_valid_df.iloc[valid_idx]
    
    return train_df, validation_df, test_df


def split_random_patient(data_df, split_ratio, seed):
    '''
        This function splits the data into train, validation and test sets using random splitting of patients.
        patients do not overlap between train, validation and test sets.
        Uses a two-step process:
        1. First split patients into train, valid and test
        2. Then get utterances from each patient
    '''
    logger.info(f"Splitting data into train, validation and test sets using random splitting of patients.")
    train_ratio, valid_ratio, test_ratio = split_ratio
    
    # Get unique patients and their labels
    patient_df = data_df.groupby('patient_id')['patient_label'].first().reset_index()
    
    # First split: (train+valid) vs test
    first_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_valid_idx, test_idx = next(first_splitter.split(patient_df, patient_df['patient_label']))
    
    train_valid_patients = patient_df.iloc[train_valid_idx]['patient_id']
    test_patients = patient_df.iloc[test_idx]['patient_id']
    
    # Second split: train vs valid
    valid_ratio_adjusted = valid_ratio / (train_ratio + valid_ratio)
    second_splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio_adjusted, random_state=seed)
    train_idx, valid_idx = next(second_splitter.split(
        patient_df.iloc[train_valid_idx], 
        patient_df.iloc[train_valid_idx]['patient_label']
    ))
    
    train_patients = train_valid_patients.iloc[train_idx]
    valid_patients = train_valid_patients.iloc[valid_idx]
    
    # Get all utterances for each split based on patient assignments
    train_df = data_df[data_df['patient_id'].isin(train_patients)]
    validation_df = data_df[data_df['patient_id'].isin(valid_patients)]
    test_df = data_df[data_df['patient_id'].isin(test_patients)]
    
    return train_df, validation_df, test_df


def get_data_df(audio_files, annotation_files, audio_processing_options):
    '''
        This function creates the dataframe with the following columns:
            - segment_id: unique identifier for each segment
            - audio_file: path to the audio file
            - start_end_time: tuple of start and end time of the segment
            - segment_duration: duration of the segment in seconds
            - utterance_label: label for the segment (normal or abnormal where abnormal is defined as either wheezes or crackles or both)
            - patient_label: label for the patient based on the majority of the utterance labels
            - patient_id: patient id for the segment
    '''
    # Initialize empty list to store rows
    data_list = []
    
    for audio_file, annotation_file in zip(audio_files, annotation_files):
        annotations_df = read_annotation_file(annotation_file)
        processed_annotations_df = chunk_segments(annotations_df, audio_processing_options)
        patient_id = os.path.basename(audio_file).split('_')[0]
        for idx, row in processed_annotations_df.iterrows():
            label = 'normal' if row['crackles'] == 0 and row['wheezes'] == 0 else 'abnormal'
            seg_index = f"{os.path.basename(audio_file).split('.')[0]}_seg{idx}"
            data_list.append({
                'segment_id': seg_index,
                'wav': audio_file,
                'start_end_time': (row['start'], row['end']),
                'segment_duration': row['segment_duration'],
                'utterance_label': label,
                'patient_id': patient_id
            })
    
    # Create DataFrame from list of dictionaries
    data_df = pd.DataFrame(data_list)
    
    if not data_df.empty:
        # Group by patient_id and get majority vote for patient label
        patient_labels = data_df.groupby('patient_id')['utterance_label'].agg(
            lambda x: x.value_counts().index[0]
        ).to_dict()
        
        # Map patient labels back to the DataFrame
        data_df['patient_label'] = data_df['patient_id'].map(patient_labels)

    return data_df


def read_annotation_file(annotation_file):
    annotations_df = pd.read_csv(annotation_file, delimiter='\t', header=None)
    annotations_df.columns = ['start', 'end', 'crackles', 'wheezes']
    annotations_df['segment_duration'] = annotations_df['end'] - annotations_df['start']
    return annotations_df


def chunk_segments(annotations_df, audio_processing_options):
    chunk_size = float(audio_processing_options.get('chunk_windows_seconds',5))  # default 5 seconds
    overlap = float(audio_processing_options.get('overlap', 0.5))  # default 50% overlap
    hop_size = chunk_size * (1 - overlap)  # time between chunk starts
    
    chunked_segments = []
    
    for idx, row in annotations_df.iterrows():
        segment_duration = row['segment_duration']  # already in seconds
        
        if segment_duration > chunk_size:
            # Calculate number of chunks needed with overlap
            num_chunks = int((segment_duration - chunk_size) // hop_size) + 1
            
            # Create overlapping chunks
            for i in range(num_chunks):
                start_time = row['start'] + (i * hop_size)
                end_time = start_time + chunk_size
                
                # Ensure we don't exceed the original segment end
                if end_time > row['end']:
                    end_time = row['end']
                
                chunked_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'crackles': row['crackles'],
                    'wheezes': row['wheezes'],
                    'segment_duration': end_time - start_time
                })
                
            # Check if we need a final chunk to cover the remaining duration
            last_end = row['start'] + (num_chunks * hop_size) + chunk_size
            if last_end < row['end']:
                start_time = row['end'] - chunk_size
                chunked_segments.append({
                    'start': start_time,
                    'end': row['end'],
                    'crackles': row['crackles'],
                    'wheezes': row['wheezes'],
                    'segment_duration': row['end'] - start_time
                })
        else:
            # Keep original segment if it's smaller than chunk_size
            chunked_segments.append(row.to_dict())
    
    return pd.DataFrame(chunked_segments)


def get_audio_sample_rate(file_path):
    """Check audio file's sample rate without loading the entire file."""
    try:
        
        info = sf.info(file_path)
        return info.samplerate
    except Exception as e:
        logger.warning(f"Could not read audio file {file_path}: {str(e)}")
        return None


if __name__ == "__main__":
    data_original = "/u/home/s/spaceman/project-voicesafe/spapl/vijay/speechbrain/cough_audio/icbi/"
    audio_dir = os.path.join(data_original, "audio_data")
    annotation_dir = os.path.join(data_original, "annotation_data")
    save_json_train = os.path.join("train.json")
    save_json_valid = os.path.join("valid.json")
    save_json_test = os.path.join("test.json")
    audio_processing_options = {'chunk_windows_seconds': 2, 'overlap': 0.75}
    split_type = 'random'
    split_variable = 'utterance'
    split_ratio = [0.7, 0.1, 0.2]
    seed = 42
    prep_icbi(audio_dir, 
              annotation_dir, 
              save_json_train, 
              save_json_valid, 
              save_json_test, 
              audio_processing_options, 
              split_type, 
              split_variable, 
              split_ratio,
              seed)