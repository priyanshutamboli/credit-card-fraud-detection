Credit Card Fraud Detection System🚀
Live Demo: https://credit-card-fraud-detection-gq67tptltar3wourc85ter.streamlit.app/

Try the live application - no installation required!A machine learning web application that detects fraudulent credit card transactions using Random Forest classification.

Project Overview
This project tackles one of the most challenging problems in financial machine learning: detecting fraudulent credit card transactions in highly imbalanced datasets. In the real world, fraudulent transactions represent only a tiny fraction of all transactions, making traditional classification approaches ineffective.

The Challenge:
Credit card fraud detection presents several unique challenges. First, the extreme class imbalance where only 0.17% of transactions are fraudulent means that a naive model predicting "legitimate" for every transaction would achieve 99.83% accuracy while catching zero fraud. Second, the cost of false negatives (missed fraud) is high in terms of financial loss and customer trust, while false positives (blocking legitimate transactions) create customer friction and dissatisfaction. Third, the features are anonymized through PCA transformation for privacy reasons, making domain knowledge less applicable.

The Solution:
This project implements a complete end-to-end machine learning pipeline specifically designed to handle imbalanced classification. The solution uses ensemble learning with Random Forest, which combines predictions from 100 decision trees to achieve robust fraud detection. Each tree is trained on random subsets of both data samples and features, reducing overfitting and improving generalization.The model employs class weighting to automatically adjust for the imbalance, making the algorithm more sensitive to the minority fraud class. Rather than relying on accuracy, the evaluation focuses on precision, recall, and F1 score, which better capture the model's ability to detect fraud while minimizing false alarms.

Key Results:
The final Random Forest model achieves an F1 score of 85.56%, successfully balancing precision and recall. With 93.90% precision, only 6% of fraud alerts are false alarms, minimizing customer inconvenience. The 78.57% recall rate means the system catches 77 out of 98 fraudulent transactions in the test set, significantly reducing financial losses.The model outperforms both baseline and single decision tree approaches. Feature importance analysis reveals that V14, V10, and V12 are the most critical features for fraud detection, together contributing nearly 40% of the model's predictive power.

Practical Application:
The system is deployed as an interactive web application using Streamlit, allowing users to upload transaction data and receive instant fraud predictions. Users can view detailed probability scores for each transaction, filter results to see only flagged frauds, and download the complete analysis as a CSV file. The application handles data preprocessing automatically, scaling features appropriately before passing them to the trained model.This project demonstrates proficiency in handling imbalanced datasets, building production-ready machine learning models, creating interactive web applications, and understanding the business implications of model predictions in a real-world context.

Model Performance
Final Model Metrics:

F1 Score: 85.56%
Precision: 93.90%
Recall: 78.57%
Accuracy: 99.95%
ROC-AUC: 0.97+
Model Comparison Results:Baseline Model (Always predicts legitimate):

Precision: 0.00%
Recall: 0.00%
F1 Score: 0.00%
Accuracy: 99.83% (misleading!)
Decision Tree:

Precision: 65.14%
Recall: 72.45%
F1 Score: 68.60%
Random Forest (Selected):

Precision: 93.90%
Recall: 78.57%
F1 Score: 85.56%

Technologies Used:
Python 3.8+: Core programming language
Scikit-learn: Machine learning models and preprocessing
Pandas and NumPy: Data manipulation and numerical operations
Streamlit: Interactive web application framework
Matplotlib and Seaborn: Data visualization
Joblib: Model serialization and deployment

