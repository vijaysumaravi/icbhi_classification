import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def read_label_encoder(label_encoder_file):
    label_map = {}
    with open(label_encoder_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '=>' in line:
                label, id_str = line.strip().split('=>')
                label = label.strip().strip("'")
                id_val = int(id_str.strip())
                label_map[label] = id_val
    return label_map

def compute_metrics(predictions_file, label_encoder_file, output_folder):
    # Read predictions CSV file
    df = pd.read_csv(predictions_file)
    
    # Load label mapping
    label_map = read_label_encoder(label_encoder_file)

    # label encoding is:  'abnormal' => 0, 'normal' => 1
    
    # Get true labels and predictions
    y_true = df['true_value'].values
    y_pred = df['prediction'].values
    

    if label_map['abnormal'] == 0:
        y_true = [1-label for label in y_true]
        y_pred = [1-label for label in y_pred]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate sensitivity (recall for positive class) and specificity (recall for negative class)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    score = (sensitivity + specificity) / 2
    
    # Compute F1-macro
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Print header and metrics as a comma-separated row
    print("F1-macro,Sensitivity,Specificity, Average Score, Confusion_Matrix(TN,FP,FN,TP)")
    print(f"{f1_macro:.4f},{sensitivity:.4f},{specificity:.4f},{score:.4f},{cm[0,0]},{cm[0,1]},{cm[1,0]},{cm[1,1]}")
    
    # Write metrics to a text file in the output folder
    metrics_file = f"{output_folder}/eval_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("F1-macro,Sensitivity,Specificity, Average Score, Confusion_Matrix(TN,FP,FN,TP)\n")
        f.write(f"{f1_macro:.4f},{sensitivity:.4f},{specificity:.4f},{score:.4f},{cm[0,0]},{cm[0,1]},{cm[1,0]},{cm[1,1]}\n")
    
    return f1_macro, sensitivity, specificity, cm

def plot_training_curves(log_file, output_folder, window_size=3):
    # Read the log file line by line
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Initialize lists to store extracted values
    epochs, train_losses, valid_losses, valid_errors = [], [], [], []

    # Process each line
    for line in lines:
        # Skip lines that contain 'loaded'
        if 'loaded' in line:
            continue

        # Split the line by commas and further split by colons to extract values
        parts = line.strip().split(', ')
        epoch = int(parts[0].split(': ')[1])  # Extract epoch number
        train_loss = float(parts[1].split(': ')[2].split()[0])  # Extract train loss
        valid_loss = float(parts[1].split(': ')[3].split()[0])  # Extract validation loss
        valid_error = float(parts[2].split(': ')[1])  # Extract validation error

        # Append to lists
        epochs.append(epoch)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_errors.append(valid_error)

    # Calculate moving averages
    train_loss_ma = [sum(train_losses[i:i+window_size])/window_size for i in range(len(train_losses)-window_size+1)]
    valid_loss_ma = [sum(valid_losses[i:i+window_size])/window_size for i in range(len(valid_losses)-window_size+1)]
    valid_error_ma = [sum(valid_errors[i:i+window_size])/window_size for i in range(len(valid_errors)-window_size+1)]
    
    # Adjust epochs to match the length of moving averages
    adjusted_epochs = epochs[window_size-1:]

    # Plot train and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(adjusted_epochs, train_loss_ma, label='Train Loss (MA)', color='blue')
    plt.plot(adjusted_epochs, valid_loss_ma, label='Validation Loss (MA)', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    
    # Plot validation error
    plt.subplot(1, 2, 2)
    plt.plot(adjusted_epochs, valid_error_ma, label='Validation Error (MA)', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Validation Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_folder}/training_curves.png")

if __name__ == "__main__":
    predictions_file = "results_utterance_split/ECAPA-TDNN/42/predictions.csv"  # Update with your predictions file path
    label_encoder_file = "results_utterance_split/ECAPA-TDNN/42/save/label_encoder.txt"  # Update with your label encoder file path
    output_folder = "results_utterance_split/ECAPA-TDNN/42"
    compute_metrics(predictions_file, label_encoder_file, output_folder)
    log_file = "results_utterance_split/ECAPA-TDNN/42/train_log.txt"  # Update with your log file path
    plot_training_curves(log_file, output_folder)