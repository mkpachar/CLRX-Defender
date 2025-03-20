import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from scipy.special import softmax
from xgboost import XGBClassifier
from torch.optim.lr_scheduler import StepLR
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# -----------------------------
# 1) FOCAL LOSS DEFINITION
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(focal_loss)

# -----------------------------
# 2) SIMPLIFIED CNN-LSTM MODEL
# -----------------------------
class SimplifiedCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimplifiedCNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(32, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# -----------------------------
# 3) LOAD AND ANALYZE DATASET
# -----------------------------
# Load the complete dataset
data = pd.read_csv('drebin.csv', low_memory=False)

# Split the dataset to use only 100%
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_subset, _, y_subset, _ = train_test_split(X, y, test_size=0.1, random_state=2024)

print("Dataset Details:")
print(f"Total samples: {len(data)}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {y.nunique()}")
print(f"Class distribution:\n{y.value_counts(normalize=True)}")

print("\nDetailed Dataset Information:")
data.info(verbose=True, show_counts=True)

print("\nSummary Statistics:")
print(data.describe())

# -----------------------------
# 4) DATA PREPROCESSING
# -----------------------------
X_subset = X_subset.replace('?', np.nan)
X_subset = X_subset.apply(pd.to_numeric, errors='coerce')
X_subset = X_subset.fillna(X_subset.mean())

selector = SelectKBest(chi2, k=215)
X_new = selector.fit_transform(X_subset, y_subset)

le = LabelEncoder()
y_subset = le.fit_transform(y_subset)

# Split the subset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y_subset, test_size=0.3, random_state=2024)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5) ADVANCED RESAMPLING
# -----------------------------
smoteenn = SMOTEENN(random_state=2024)
X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train_scaled, y_train)

# OPTIONAL DATA AUGMENTATION:
minority_index = (y_train_resampled == 1)
X_minority = X_train_resampled[minority_index]
y_minority = y_train_resampled[minority_index]
X_aug, y_aug = resample(
    X_minority,
    y_minority,
    replace=True,
    n_samples=int(len(y_minority) * 0.3),
    random_state=2024
)
X_train_final = np.vstack([X_train_resampled, X_aug])
y_train_final = np.hstack([y_train_resampled, y_aug])

X_train, X_val, y_train, y_val = train_test_split(X_train_final, y_train_final, test_size=0.2, random_state=2024)

print("\nProcessed Dataset Details:")
print(f"Number of features after selection: {X_new.shape[1]}")
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")
print("-----------------------------")

# -----------------------------
# 6) TORCH DATASETS & LOADERS
# -----------------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# -----------------------------
# 7) CNN-LSTM TRAINING SETUP
# -----------------------------
model = SimplifiedCNNLSTM(X_train.shape[1], 128, len(np.unique(y))).to('cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
focal_loss = FocalLoss()
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = 30
patience = 5
best_val_f1 = 0
epochs_without_improvement = 0
best_model = None

train_losses = []
val_losses = []
# -----------------------------
# 8) TRAIN LOOP WITH EARLY STOP
# -----------------------------
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = focal_loss(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss/len(train_loader)
    train_losses.append(avg_train_loss)

    scheduler.step()

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_pred = []
    val_true = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = focal_loss(outputs, batch_y)
            val_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            val_pred.extend(predictions.numpy())
            val_true.extend(batch_y.numpy())
    
    avg_val_loss = val_loss/len(val_loader)
    val_losses.append(avg_val_loss)

    val_f1 = f1_score(val_true, val_pred, average='weighted')
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model = model.state_dict()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")
    print("-----------------------------")

    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, 'orange', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Validation loss')
plt.legend()
plt.grid(True)
plt.show()

model.load_state_dict(best_model)
model.eval()

# -----------------------------
# 9) CNN-LSTM PREDICTIONS
# -----------------------------
y_pred = []
y_true = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predictions = torch.max(outputs, 1)
        y_pred.extend(predictions.numpy())
        y_true.extend(batch_y.numpy())

cnn_lstm_reshaped = np.array(y_pred).reshape(-1, 1)
cnn_lstm_probabilities = softmax(cnn_lstm_reshaped, axis=1)[:, 0]

# -----------------------------
# 10) RANDOM FOREST HYPERPARAM TUNING
# -----------------------------
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_base = RandomForestClassifier(random_state=2024, class_weight='balanced')
grid_search_rf = GridSearchCV(estimator=rf_base, param_grid=param_grid_rf, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf_params = grid_search_rf.best_params_
print("\nBest Random Forest Parameters:", best_rf_params)

rf_model = RandomForestClassifier(**best_rf_params, random_state=2024, class_weight='balanced')
rf_model.fit(X_train, y_train)
rf_probabilities = rf_model.predict_proba(X_test_scaled)[:, 1]

# -----------------------------
# 11) XGBOOST MODEL
# -----------------------------
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=2024)
xgb_model.fit(X_train, y_train)
xgb_probabilities = xgb_model.predict_proba(X_test_scaled)[:, 1]

# -----------------------------
# 12) FINAL ENSEMBLE
# -----------------------------
ensemble_prob = (
    0.3 * cnn_lstm_probabilities +
    0.35 * rf_probabilities +
    0.35 * xgb_probabilities
)
final_predictions = (ensemble_prob > 0.5).astype(int)

# -----------------------------
# 13) EVALUATE FINAL PERFORMANCE
# -----------------------------
accuracy = accuracy_score(y_true, final_predictions)
precision = precision_score(y_true, final_predictions, average='weighted')
recall = recall_score(y_true, final_predictions, average='weighted')
f1 = f1_score(y_true, final_predictions, average='weighted')
auc = roc_auc_score(y_true, ensemble_prob)

# Calculate confusion matrix
cm = confusion_matrix(y_true, final_predictions)
tn, fp, fn, tp = cm.ravel()

print("\nFinal Ensemble Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC Score: {auc:.4f}")
print(f"True Negatives (TN): {tn}")
print(f"True Positives (TP): {tp}")
print(f"False Negatives (FN): {fn}")
print(f"False Positives (FP): {fp}")

# Calculate additional metrics
specificity = tn / (tn + fp)
npv = tn / (tn + fn)  # Negative Predictive Value
fpr = fp / (fp + tn)  # False Positive Rate
fnr = fn / (fn + tp)  # False Negative Rate

print(f"Specificity: {specificity:.4f}")
print(f"Negative Predictive Value: {npv:.4f}")
print(f"False Positive Rate: {fpr:.4f}")
print(f"False Negative Rate: {fnr:.4f}")
# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, ensemble_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
