import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset_path = "."  


activity_labels = pd.read_csv(os.path.join(dataset_path, "activity_labels.txt"), sep=" ", header=None, names=["label", "activity"])

X_train = pd.read_csv(os.path.join(dataset_path, "train/X_train.txt"), sep="\s+", header=None)
y_train = pd.read_csv(os.path.join(dataset_path, "train/y_train.txt"), sep="\s+", header=None, names=["label"])
X_test = pd.read_csv(os.path.join(dataset_path, "test/X_test.txt"), sep="\s+", header=None)
y_test = pd.read_csv(os.path.join(dataset_path, "test/y_test.txt"), sep="\s+", header=None, names=["label"])

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data["label"] = train_data["label"].map(activity_labels.set_index("label")["activity"])
test_data["label"] = test_data["label"].map(activity_labels.set_index("label")["activity"])

# Defining the classes to keep
classes_to_keep = {
    "WALKING": "walking",
    "WALKING_UPSTAIRS": "walking_upstairs",
    "WALKING_DOWNSTAIRS": "walking_downstairs",
    "SITTING": "sitting",
    "STANDING": "standing",
    "LAYING": "laying"
}

train_data = train_data[train_data["label"].isin(classes_to_keep.keys())]
test_data = test_data[test_data["label"].isin(classes_to_keep.keys())]

train_data["label"] = train_data["label"].replace(classes_to_keep)
test_data["label"] = test_data["label"].replace(classes_to_keep)



print("Initial class distribution (Train):\n", train_data["label"].value_counts())
print("Initial class distribution (Test):\n", test_data["label"].value_counts())


# Normalizing the sensor data 
scaler = StandardScaler()
sensor_columns = [col for col in train_data.columns if col != "label"]
train_data[sensor_columns] = scaler.fit_transform(train_data[sensor_columns])
test_data[sensor_columns] = scaler.transform(test_data[sensor_columns])


train_output_path = os.path.join(dataset_path, "uci_har_train_preprocessed.csv")
test_output_path = os.path.join(dataset_path, "uci_har_test_preprocessed.csv")
train_data.to_csv(train_output_path, index=False)
test_data.to_csv(test_output_path, index=False)

print(f"Preprocessing complete! Preprocessed data saved to: {train_output_path} and {test_output_path}")
print(f"Final dataset shape (Train): {train_data.shape}")
print(f"Final dataset shape (Test): {test_data.shape}")
print(f"Class distribution (Train):\n{train_data['label'].value_counts()}")
print(f"Class distribution (Test):\n{test_data['label'].value_counts()}")