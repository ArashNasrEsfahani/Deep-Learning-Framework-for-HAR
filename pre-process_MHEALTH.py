import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


dataset_path = "."  

columns = [
    "chest_acc_x", "chest_acc_y", "chest_acc_z",  # Chest acceleration
    "ecg_1", "ecg_2", 
    "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",  # Left ankle acceleration 
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",  # Left ankle gyroscope 
    "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",  # Left ankle magnetometer 
    "arm_acc_x", "arm_acc_y", "arm_acc_z",  # Right lower arm acceleration 
    "arm_gyro_x", "arm_gyro_y", "arm_gyro_z",  # Right lower arm gyroscope 
    "arm_mag_x", "arm_mag_y", "arm_mag_z",  # Right lower arm magnetometer 
    "label"  # Activity label
]

# Defining the classes to keep 
classes_to_keep = {
    1: "standing_sitting", 
    4: "walking",          
    10: "jogging",       
    11: "running"           
}


all_data = []

for subject_id in range(1, 11):  
    file_path = os.path.join(dataset_path, f"mHealth_subject{subject_id}.log")
    
    df = pd.read_csv(file_path, sep="\t", header=None, names=columns)

    df = df[df["label"].isin(list(classes_to_keep.keys()) + [0])]

    df["label"] = df["label"].replace(classes_to_keep)  
    df["label"] = df["label"].replace({1: "standing_sitting", 2: "standing_sitting"})
    
    all_data.append(df)

combined_data = pd.concat(all_data, ignore_index=True)
print("Initial class distribution:\n", combined_data["label"].value_counts())


# Balancing the dataset
class_counts = combined_data["label"].value_counts()
min_samples = class_counts.min() if not class_counts.empty else 0

balanced_data = []
for class_name in ["standing_sitting", "walking", "jogging", "running", 0]:
    class_data = combined_data[combined_data["label"] == class_name]
    if len(class_data) > 0:  
        balanced_data.append(class_data.sample(n=min_samples, random_state=42))

balanced_data = pd.concat(balanced_data, ignore_index=True)

# Normalizing the sensor data 
scaler = StandardScaler()
sensor_columns = [col for col in balanced_data.columns if col != "label"]
balanced_data[sensor_columns] = scaler.fit_transform(balanced_data[sensor_columns])

output_path = os.path.join(dataset_path, "mhealth_balanced_preprocessed.csv")
balanced_data.to_csv(output_path, index=False)

print(f"Preprocessing complete! Preprocessed data saved to: {output_path}")
print(f"Final dataset shape: {balanced_data.shape}")
print(f"Class distribution:\n{balanced_data['label'].value_counts()}")