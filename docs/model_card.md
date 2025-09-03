# Model Card: TabNet for Emergency Department Unscheduled Return Visit Prediction

## Model Overview

**Model Name:** TabNet-ED-URV-Predictor  
**Model Version:** 1.0  
**Model Type:** Deep Learning (TabNet Architecture)  
**Publication:** Performance Comparison of Artificial Intelligence Models in Predicting 72-hour Emergency Department Unscheduled Return Visits  
**Development Date:** 2024  
**Last Updated:** January 2025

## Intended Use

### Primary Use Cases
This model is designed to support clinical decision-making in emergency department settings by identifying patients at elevated risk for unscheduled return visits within 72 hours of initial discharge. The prediction tool enables healthcare providers to implement targeted interventions including enhanced discharge planning, comprehensive medication counseling, and structured follow-up arrangements for high-risk patients.

### Target Population
Adult patients (≥18 years) presenting to emergency departments for internal medicine conditions who are discharged home without hospital admission. The model was developed and validated using data from a tertiary academic medical center serving an urban population in Shanghai, China.

### Clinical Application Context
The model functions as a clinical decision support tool integrated into routine emergency department workflow. Risk stratification occurs during the discharge planning phase using readily available clinical variables documented during standard patient evaluation procedures.

## Model Architecture and Training

### Technical Specifications
- **Architecture:** TabNet with sequential attention mechanisms and learnable feature selection masks
- **Input Features:** 25 clinical variables encompassing demographic, clinical, behavioral, and administrative characteristics
- **Training Dataset:** 100,235 patient encounters (70% of total cohort)
- **Validation Method:** 5-fold cross-validation with Bayesian optimization for hyperparameter tuning
- **Class Balancing:** Hybrid sampling approach utilizing ADASYN oversampling and TomekLinks undersampling

### Training Data Characteristics
- **Total Cohort Size:** 143,192 emergency department visits
- **URV Prevalence:** 16.8% (24,117 cases)
- **Study Period:** January 1 – December 31, 2023
- **Institution:** Shanghai East Hospital, Tongji University
- **Missing Data Rate:** 6.8% overall (handled via complete case analysis)

## Performance Metrics

### Primary Performance Indicators
- **AUROC:** 0.867 (95% CI: 0.854-0.880)
- **Sensitivity:** 0.809 (95% CI: 0.801-0.816)
- **Specificity:** 0.745 (95% CI: 0.739-0.751)
- **Accuracy:** 0.756 (95% CI: 0.751-0.760)
- **F1-Score:** 0.528 (95% CI: 0.522-0.535)
- **Precision:** 0.392 (95% CI: 0.385-0.400)

### Calibration and Clinical Utility
The model demonstrates acceptable calibration performance across probability ranges with sustained clinical utility for threshold probabilities between 10-30% as assessed by decision curve analysis. Net benefit analysis confirms clinical value compared to treat-all or treat-none strategies within this probability range.

### Comparative Performance
TabNet outperformed traditional machine learning approaches including logistic regression (AUROC 0.838), random forest (AUROC 0.852), support vector machine (AUROC 0.835), and XGBoost (AUROC 0.621) in head-to-head comparison using identical datasets and evaluation protocols.

## Key Predictive Features

### Primary Risk Factors (Ranked by Importance)
1. **Initial Diagnosis - Digestive System Diseases**
2. **Initial Diagnosis - Respiratory System Diseases**
3. **Patient Age**
4. **Triage Classification (P3 Level)**
5. **Emergency Department Visit Frequency (Previous Month)**

### Feature Interpretation
Ablation studies confirm that digestive system diagnoses represent the most influential predictor, with model performance degrading from accuracy 0.756 to 0.643 upon feature removal. The combination of respiratory and digestive system diagnoses, patient age, intermediate triage acuity, and recent healthcare utilization patterns collectively account for the majority of predictive capability.

## Limitations and Constraints

### Methodological Limitations
This model represents a single-center retrospective analysis with inherent generalizability constraints. External validation across diverse healthcare systems, patient populations, and geographic regions remains necessary before widespread clinical deployment. The model cannot capture psychological factors, physician-patient communication quality, or unmeasured social determinants that influence return visit behavior.

### Technical Constraints
- **Computational Requirements:** Requires adequate infrastructure for real-time inference during high-volume clinical operations
- **Data Dependencies:** Model performance assumes consistent variable coding and data capture protocols
- **Temporal Stability:** Potential degradation due to population drift or changes in clinical practice patterns over time

### Clinical Application Boundaries
The model specifically applies to adult internal medicine patients and excludes pediatric, surgical, and psychiatric presentations. Performance characteristics may not generalize to healthcare systems with different triage protocols, staffing patterns, or patient care pathways.

## Ethical Considerations and Fairness

### Algorithmic Fairness Assessment
Subgroup analysis across age categories, triage levels, and diagnostic classifications demonstrates consistent recall performance (0.799-0.830) with coefficient of variation within acceptable limits for clinical deployment. Age-stratified analysis reveals accuracy variation from 0.839 in younger adults to 0.664 in elderly patients, representing clinically acceptable performance differentials.

### Privacy and Data Protection
Model development adhered to institutional review board protocols with appropriate ethical oversight. All training data underwent de-identification procedures consistent with medical research privacy standards. The synthetic dataset provided for reproducibility maintains statistical fidelity while ensuring complete patient privacy protection.

### Potential for Bias
Healthcare utilization patterns may reflect underlying social, economic, or cultural factors that could introduce systematic bias in prediction accuracy across different patient populations. Regular fairness audits should accompany clinical deployment to monitor for disparate impact across protected demographic groups.

## Risk Assessment and Mitigation

### Clinical Risks
- **False Negatives:** Missed high-risk patients may experience adverse outcomes during undetected return visits
- **False Positives:** Unnecessary interventions may increase healthcare costs and patient anxiety
- **Over-reliance:** Clinical judgment should remain paramount in discharge decision-making

### Mitigation Strategies
Implementation should include clinical override mechanisms, regular model performance monitoring, and integration with existing quality improvement protocols. Healthcare providers require training in model interpretation and appropriate clinical response to risk stratification outputs.

## Governance and Maintenance

### Model Monitoring
Continuous performance surveillance should track prediction accuracy, calibration drift, and fairness metrics across patient subgroups. Monthly evaluation of key performance indicators enables timely detection of model degradation requiring retraining or recalibration.

### Update Protocols
Annual model retraining using updated patient cohorts maintains prediction accuracy as clinical practices and patient populations evolve. Version control procedures ensure reproducibility and enable rollback capabilities if performance deterioration occurs.

## Contact Information

**Corresponding Authors:**
- Chongjun Fan, PhD (cjfan@usst.edu.cn)
- Honglin Xiong, PhD (xionghl@sumhs.edu.cn)

**Institutional Affiliation:**
Shanghai East Hospital, Tongji University School of Medicine  
Shanghai University of Medicine & Health Sciences

## Disclaimer

This model serves as a clinical decision support tool and should not replace professional medical judgment. Healthcare providers maintain ultimate responsibility for patient care decisions and discharge planning. The model requires external validation before implementation in clinical environments different from the development setting.
