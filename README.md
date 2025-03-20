CLRX Defender: A Hybrid CNN-LSTM and Ensemble Learning Framework for Advanced Android Malware Detection
Overview
The CLRX Defender is a cutting-edge Android malware detection system that leverages a hybrid approach combining Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) networks, Random Forest, and XGBoost. This ensemble model is designed to enhance the security of Android devices by effectively identifying and classifying malicious applications.
Key Features
•	Hybrid Architecture: Combines CNN-LSTM for sequential pattern recognition, Random Forest for feature importance, and XGBoost for gradient-boosted decision boundaries.
•	High Accuracy: Achieves an impressive accuracy of 98.10% on the Drebin dataset.
•	Efficient Handling of Class Imbalance: Utilizes SMOTEENN for resampling to manage class imbalance and prevent overfitting.
•	Interpretability: Incorporates SHAP-compatible feature engineering for model transparency.
Getting Started
1.	Clone the Repository:
git clone https://github.com/your-username/CLRX-Defender.git

2.	Install Dependencies:
Ensure you have Python installed. Then, install necessary packages using pip:
pip install -r requirements.txt

3.	Run the Model:
Execute the main script to train and test the model:
python CNN-LSTM_RF_XGBoostDrebin.py

Dataset
•	Drebin Dataset: The model uses a comprehensive dataset of 15,036 Android applications.
•	Features: Extracts 215 features including API calls, permissions, and system actions.
Performance Metrics
Metric	Value
Accuracy	98.10%
Precision	98.13%
Recall	98.10%
F1 Score	98.11%
AUC Score	99.67%
Specificity	97.75%
False Negative Rate	1.31%

Contributing
Contributions are welcome! Please submit pull requests with detailed explanations of changes.
Acknowledgments
•	Authors: Mandeep Kumar and Abhishek Kajal
•	Institution: Department of CSE, GJUS&T, Hisar, India
 
This README provides essential information for users to understand and contribute to the CLRX Defender project.
