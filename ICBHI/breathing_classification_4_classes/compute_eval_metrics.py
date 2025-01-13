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
    
    # Get true labels and predictions
    y_true = df['true_value'].values
    y_pred = df['prediction'].values
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class metrics
    # Compute sensitivity (recall) for each class
    sensitivities = []
    specificities = []
    for i in range(4):  # For each class
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    # Average metrics
    avg_sensitivity = sum(sensitivities) / 4
    avg_specificity = sum(specificities) / 4
    score = (avg_sensitivity + avg_specificity) / 2
    
    # Compute F1-macro
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Print header and metrics
    print("F1-macro,Avg-Sensitivity,Avg-Specificity,Average-Score")
    print(f"{f1_macro:.4f},{avg_sensitivity:.4f},{avg_specificity:.4f},{score:.4f}")
    
    # Print per-class metrics
    print("\nPer-class metrics:")
    print("Class,Sensitivity,Specificity")
    class_names = ['normal', 'crackles', 'wheezes', 'both']
    for i in range(4):
        print(f"{class_names[i]},{sensitivities[i]:.4f},{specificities[i]:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm)
    
    # Write metrics to a text file
    metrics_file = f"{output_folder}/eval_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("F1-macro,Avg-Sensitivity,Avg-Specificity,Average-Score\n")
        f.write(f"{f1_macro:.4f},{avg_sensitivity:.4f},{avg_specificity:.4f},{score:.4f}\n\n")
        
        f.write("Per-class metrics:\n")
        f.write("Class,Sensitivity,Specificity\n")
        for i in range(4):
            f.write(f"{class_names[i]},{sensitivities[i]:.4f},{specificities[i]:.4f}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    
    return f1_macro, avg_sensitivity, avg_specificity, cm

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
    predictions_file = "results_patient_split/ECAPA-TDNN/42/predictions.csv"  # Update with your predictions file path
    label_encoder_file = "results_patient_split/ECAPA-TDNN/42/save/label_encoder.txt"  # Update with your label encoder file path
    output_folder = "results_patient_split/ECAPA-TDNN/42"
    compute_metrics(predictions_file, label_encoder_file, output_folder)
    log_file = "results_patient_split/ECAPA-TDNN/42/train_log.txt"  # Update with your log file path
    plot_training_curves(log_file, output_folder)