# AUTISM-SPECTRUM-DISORDER-PREDICTION-USING-ADABOOST-CLASSIFIER

This project focuses on analyzing the Toddler Autism dataset to better understand the factors that may indicate Autism Spectrum Disorder (ASD) traits in toddlers. The workflow begins with exploratory data analysis (EDA), where various visualizations such as count plots, pie charts, and bar charts are used to uncover patterns in demographic features (e.g., sex, ethnicity, age), medical history (e.g., jaundice, family history of ASD), and responses to the standardized Q-CHAT-10 screening questionnaire.

**Project Overview** :
- Performed exploratory data analysis (EDA) with visualizations to understand the distribution of ASD cases across various demographic and medical factors.
- Built an AdaBoost Classifier to predict ASD traits.
- Evaluated the model with metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC curve.

**Dataset** :
- Name: Toddler Autism dataset
- Source: UCI Machine Learning Repository - Autism Screening Adult, Adolescent, and Toddler Data Sets
- Target Column: Class/ASD Traits (Yes / No)
- Features include:
   - Demographics (Sex, Age, Ethnicity, etc.)
   - Medical history (Jaundice, Family history of ASD)
   - Responses to Q-CHAT-10 screening questions

**Exploratory Data Analysis (EDA)** :
- Count plots for categorical features
- Pie charts for ASD distribution and ethnicity distribution
- Bar charts for ASD cases by:
    - Ethnicity
    - Sex
    - Jaundice
    - Family history of ASD
    - Age & QCHAT-10 score
- Feature-wise distribution by ASD class

**Model** :
- Algorithm: AdaBoost Classifier (sklearn.ensemble.AdaBoostClassifier)
- Trained on scaled features.
- Predicted ASD traits on test set.

**Results** :

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.88  |
| Precision | 0.85  |
| Recall    | 0.90  |
| F1 Score  | 0.87  |


**How to Run the Project** :
1. Clone the repository :

   ```bash
   git clone https://github.com/faheem-445/AUTISM-SPECTRUM-DISORDER-PREDICTION-USING-ADABOOST-CLASSIFIER-.git
   cd AUTISM-SPECTRUM-DISORDER-PREDICTION-USING-ADABOOST-CLASSIFIER-
   ```

2. Install dependencies :

   ```bash
   pip install -r requirements.txt
     ```

3. Run the python script :

   ```bash
   python asd_adaboost.py   
   ```




