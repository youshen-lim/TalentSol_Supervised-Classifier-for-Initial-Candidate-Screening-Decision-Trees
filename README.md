# TalentSol - Applicant Tracking System Learning Project (Machine Learning Component)

This repository contains the machine learning component of the TalentSol project. It trains supervised classifiers to prioritize candidates for a job role and description. This learning project compares text embedding techniques and ensemble methods on a fixed template Kaggle recruitment dataset.

> **Important: the saved model uses Ethnicity.** The saved XGBoost pipeline one hot encodes an Ethnicity column as an input. The Ethnicity values in the training data are mostly national origin labels, such as Chinese, Irish and Nigerian. Using this input to screen candidates is not acceptable for real hiring. I plan to retrain the model without Ethnicity, Race1 and Race2. Race1 and Race2 are already excluded from the saved model. The reported metrics must be measured again after retraining. This model is a learning artifact. Do not use it for real hiring decisions. See [Fairness](#fairness) and [Limitations](#limitations).

## Project Goal

The primary goal of this project is to train a supervised classifier that can predict which job applicants are the "Best Match" for a given job role and description, thereby helping recruiters prioritize candidates during the screening process. A key focus is achieving high recall to minimize false negatives (missing out on potentially good candidates) while maintaining reasonable precision.

## Dataset

The project uses the `job_applicant_dataset.csv` dataset, which contains information about job applicants, their resumes, and whether they were considered a "Best Match" for a specific job role and description.

**Dataset Source:** The dataset is sourced from Kaggle: [Recruitment Dataset](https://www.kaggle.com/datasets/surendra365/recruitement-dataset)

The resume text in this dataset follows a fixed template, for example "Proficient in ..., with senior-level experience in the field." I have not confirmed whether the dataset is synthetic. I have not confirmed which license applies to it.

## Preprocessing and Feature Engineering

Before we can train a machine learning model on text data like job descriptions and resumes, we need to convert the words and phrases into a numerical format that the model can understand. This process is called **Feature Engineering**, where we create meaningful numerical features from the raw data.

The following preprocessing and feature engineering steps were performed on the dataset:

1.  **Data Loading:** The `job_applicant_dataset.csv` file was loaded into a pandas DataFrame.
2.  **Identifier and Sensitive Feature Removal:** I removed 'Job Applicant Name', 'Job Applicant ID', 'Age', and 'Gender' early in the preprocessing pipeline. These columns are not direct model inputs. This step does not make the data private. The resume text stays in the model input and in the committed training CSV, and resume text can contain personal information. They were also not considered relevant or appropriate for the prioritization task in this version of the project, as the focus is on text matching between the resume/CV and the job description/job role.
3.  **Race Columns:** The Race column was split into Race1 and Race2, and then both columns were dropped. Neither is a model input.
4.  **Missing Value Handling:** Checked for and confirmed no missing data points were present after initial cleaning.
5.  **Oversampling:** The classes were close to balanced, with 5,150 rows that were not a "Best Match" and 4,850 rows that were. I oversampled the "Best Match" class to 5,150 rows by sampling with replacement. This bootstrap sample duplicated some positive rows and left out others. The committed training CSV has 10,300 rows, of which 8,260 are unique. It contains 3,110 unique positive rows, so 1,740 of the 4,850 original positive rows are absent.
6.  **Categorical Encoding:** Job Roles and Ethnicity were one hot encoded. Race1 and Race2 were dropped before encoding. The saved model still uses Ethnicity as an input.
7.  **Text Feature Embedding:** This is where we convert the text from 'Job Description' and 'Resume' into numbers. We explored different methods:
    *   **CountVectorizer:** Creates a vocabulary of all unique words and counts their occurrences in each document.
    *   **TF-IDF Vectorization (Term Frequency-Inverse Document Frequency):** Calculates a score for each word based on its frequency within a document and its rarity across all documents.
    *   **HashingVectorizer:** A memory-efficient vectorizer that maps text features to a fixed-size vector using hashing.
    *   **Word2Vec Embeddings:** Learns dense numerical vector representations for words, capturing semantic relationships. Document vectors were created by averaging word vectors.
    *   **FastText Embeddings:** An extension of Word2Vec that considers sub-word information, useful for out-of-vocabulary words. Document vectors were created by averaging word vectors.

## Modeling and Evaluation

Various Decision Tree-based models and ensemble methods were trained and evaluated using **5-fold Stratified Cross-Validation** on the upsampled training data. This cross-validation approach ensures that each fold has a representative proportion of both classes ('Best Match' and 'Not Best Match'). The oversampled data contains duplicate rows. Copies of one row can fall in both a training fold and a validation fold, so the cross validation scores are likely optimistic.

Hyperparameter tuning was performed using **Grid Search Cross-Validation (`GridSearchCV`)** for individual models and some ensembles. For more complex models, parameter grids were sometimes reduced to manage computational requirements.

We explored and tuned the following model configurations:

1.  **Decision Tree with CountVectorizer:** Trained and tuned a Decision Tree model using CountVectorizer for text features.
2.  **Decision Tree with TF-IDF:** Trained and tuned a Decision Tree model using TF-IDF for text features.
3.  **Decision Tree with HashingVectorizer:** Trained and tuned a Decision Tree model using HashingVectorizer for text features.
4.  **Decision Tree with Word2Vec Embeddings:** Trained and tuned a Decision Tree model using Word2Vec embeddings.
5.  **Decision Tree with FastText Embeddings:** Trained and tuned a Decision Tree model using FastText embeddings.
6.  **Averaging Probabilities Ensemble:** An ensemble that averages the predicted probabilities from the five individual Decision Tree models (CountVectorizer, TF-IDF, HashingVectorizer, Word2Vec, FastText).
7.  **Voting Classifier Ensemble:** A Voting Classifier ensemble combining selected base estimators (individual Decision Tree models).
8.  **Stacking Classifier Ensemble:** A Stacking Classifier ensemble using selected base estimators and a Logistic Regression meta-classifier.
9.  **Random Forest Ensemble:** A Random Forest classifier using HashingVectorizer for text features.
10. **Extra Trees Ensemble:** An Extra Trees classifier using HashingVectorizer for text features.
11. **Gradient Boosting Classifier Ensemble:** A Gradient Boosting classifier using HashingVectorizer for text features.
12. **XGBoost Classifier Ensemble:** An XGBoost classifier using HashingVectorizer for text features.
13. **LightGBM Classifier Ensemble:** A LightGBM classifier using HashingVectorizer for text features.
14. **AdaBoost Decision Tree Ensemble:** An AdaBoost classifier using a Decision Tree stump as the base estimator and HashingVectorizer for text features.
15. **Hybrid Voting Classifier Ensemble:** A Voting Classifier combining predictions from a selection of the implemented individual models and ensemble methods. Word2Vec and FastText embedded models were excluded due to issues with out-of-vocabulary words during cross-validation.
16. **Hybrid Stacking Ensemble:** A Stacking Classifier combining predictions from a selection of the implemented individual models and ensemble methods with a Logistic Regression meta-classifier. Word2Vec and FastText embedded models were excluded due to issues with out-of-vocabulary words during cross-validation.

I evaluated the models with Accuracy, Precision, Recall, F1 score, and ROC AUC on a 20 percent test split. I drew the split after oversampling. Some test rows duplicate training rows, so the split is not independent and the results are likely optimistic. I used out of fold training predictions to create precision recall curves and tune thresholds toward a recall target of 0.700. I also plotted confusion matrices at the tuned thresholds.

**Methodology for Prediction (How the Model Makes a Prediction):**

For a new applicant, the prediction process involves: applying the *same* preprocessing and feature engineering steps (using the fitted transformers from the chosen trained pipeline), scaling the features, feeding the scaled features to the trained classifier to get a probability score, and finally applying the **optimized decision threshold** (derived during evaluation) to classify the applicant as 'Best Match' (1) or 'Not Best Match' (0). For ensemble models, this process involves obtaining probabilities from the base models and combining them according to the ensemble strategy before applying the optimized threshold.

**Responsible AI Considerations:**

I removed Age, Gender, Race1 and Race2 before training. The saved model still uses Ethnicity as an input. Using Ethnicity to screen candidates is not acceptable for real hiring. I plan to retrain the model without Ethnicity, Race1 and Race2 and report the new results. The metrics below must be measured again after that retrain. Further steps for a production system would include bias audits, fairness metrics, human oversight, and transparency.

### Why I am removing protected attributes from the model

I decided that protected characteristics must not be inputs to a screening model. This covers ethnicity, race, national origin, age and gender. This is a place where human judgment should override a model. A model can find a statistical signal in any column. That signal does not make the column a fair basis for a hiring decision. The current saved model makes Ethnicity, represented by national origin labels, a factor in candidate scoring. Removing Ethnicity requires a retrain, and that retrain has not happened yet. This repository trains the model. The [TalentSol ATS repository](https://github.com/youshen-lim/TalentSol---Applicant-Tracking-System-Application) serves the same model file.

## Results

The models were evaluated at decision thresholds tuned to reach about 70% recall on out of fold predictions from the upsampled training data. A comparison was then made on the 20 percent test split, which I drew after oversampling, using both default (0.5) and optimized thresholds. All results below come from the Kaggle dataset and must be measured again after retraining without Ethnicity. Key findings include:

*   **Impact of Threshold Optimization:** Changing the decision threshold changed precision and recall. In the notebook results, increases in recall often reduced precision.
*   **Target Recall Achievement:** 10 out of the 16 evaluated models achieved or exceeded the target recall of 70% on the test split when using their optimized thresholds.
*   **Selected Model:** I selected the **XGBoost Classifier Ensemble** for the saved pipeline. On the test split at its tuned threshold, it reached recall 0.703 and precision 0.567. It tied with the Hybrid Stacking Ensemble for the highest precision among models that met the recall target. I did not measure inference speed. By F1 score, the notebook ranks the HashingVectorizer Decision Tree first, with recall 0.819 and precision 0.526.
*   **Hybrid Stacking Ensemble Performance:** The **Hybrid Stacking Ensemble** reached recall 0.701 and precision 0.567 on the test split. It tied with XGBoost for the highest precision among models that met the recall target.
*   **HashingVectorizer Ensemble Performance:** The **Extra Trees Decision Tree Ensemble** and **Random Forest Decision Tree Ensemble** both use HashingVectorizer for text preprocessing. At their optimized thresholds on the test split, Extra Trees reached recall 0.728 and precision 0.525, and Random Forest reached recall 0.731 and precision 0.533.
*   **Gradient Boosting Models:** On the test split at their tuned thresholds, LightGBM reached recall 0.718 and precision 0.560. Gradient Boosting reached recall 0.709 and precision 0.542.
*   **AdaBoost Performance:** At the default threshold, AdaBoost reached recall 0.605 and precision 0.557. At its tuned threshold, it reached recall 0.699 and precision 0.548 on the test split, just below the target.
*   **Averaging vs. Voting/Stacking:** The Averaging ensemble reached recall 0.689 and precision 0.547 on the test split. The Hybrid Voting ensemble reached recall 0.698 and precision 0.551. Both fell below the recall target of 0.700.
*   **Word Embeddings vs. Vectorizers:** The CountVectorizer, HashingVectorizer and Word2Vec single trees met the recall target on the test split. The TF-IDF tree (recall 0.200) and the FastText tree (recall 0.464) did not.
*   **Models at the Recall Target:** XGBoost and Hybrid Stacking reached the highest precision, 0.567, among models that met the recall target of 0.700 on the test split. Extra Trees reached recall 0.728 and precision 0.525. Random Forest reached recall 0.731 and precision 0.533. Gradient Boosting and LightGBM also met the recall target. These results come from a split made after oversampling.
*   **Multi-view Learning:** Multi-view Learning was explored theoretically, identifying text and categorical data as potential views and discussing fusion strategies, benefits, and challenges, concluding that a full implementation was outside the current scope but laid the theoretical foundation.

### Conceptual Strengths of Decision Tree Models

Decision Tree models and their ensembles are well-suited for this text matching task due to several conceptual strengths:

*   **Handling Non-linear Relationships and Feature Interactions:** They naturally capture complex, non-linear relationships and interactions between features without assuming linear boundaries.
*   **Irrelevant Features:** This project did not test model robustness to irrelevant or noisy features.
*   **Interpretability (for individual trees):** Individual trees are highly interpretable, providing rule sets for predictions.
*   **No Assumptions about Data Distribution:** They are non-parametric, suitable for complex feature distributions like text and categorical data.
*   **Engineering Comparison:** This project did not measure engineering effort, training cost, inference cost, or memory use against neural network models.

## Decision Trees and Data Samples

Decision Trees require numerical input. Text data is converted via vectorization/embedding (CountVectorizer, TF-IDF, HashingVectorizer, Word2Vec, FastText) and categorical data via encoding (One-Hot Encoding). Trees work by recursively splitting the data based on features that best reduce impurity (e.g., using Gini Impurity or Entropy).

## Why Some Models Fail to Meet the Target Recall

Even with optimized thresholds, some models may not reach the target recall on the test split due to:

*   **Differences in Data Distribution:** The test set may have a slightly different distribution than the training data used for optimization.
*   **Model Limitations:** Some models may intrinsically be unable to achieve high recall with reasonable precision on this dataset.
*   **Overfitting:** While mitigated by OOF predictions, slight overfitting to training splits can occur.
*   **Inherent Problem Difficulty:** The task of matching resumes to job descriptions is complex, potentially limiting achievable performance regardless of the model.

## Considerations for Alternative Pruning Techniques

Standard Decision Trees use pre-pruning (e.g., `max_depth`, `min_samples_split`, `min_samples_leaf`). Post-pruning techniques like Cost-Complexity Pruning (`ccp_alpha`) are also available and can be tuned via GridSearchCV. Ensemble methods primarily regularize through aggregation, but tuning base learner complexity remains important.

## Saved Artifacts

The following artifacts from the modeling process have been saved:

*   `upsampled_training_data.csv`: The final upsampled dataset used for training. It has the columns Ethnicity, Resume, Job Roles, Job Description and Best Match.
*   `best_performing_model_pipeline.joblib`: The trained scikit-learn Pipeline object for the **XGBoost Classifier Decision Tree Ensemble**. Its inputs are Job Description, Resume, Job Roles and Ethnicity. The notebook save cell skipped saving in the committed run, and this file was committed separately. I have not confirmed that it came from the same run as the reported metrics.

## Inference and Integration

To use the trained model for predicting on new job applicants and integrating into a system:

1.  **Load the Trained Pipeline:** Load the `best_performing_model_pipeline.joblib` file using `joblib.load()`.
2.  **Predict:** Apply the loaded pipeline's `predict_proba()` method to new, unseen applicant data (after ensuring it has the expected column structure). The output probabilities can then be thresholded using the optimized decision threshold found for the XGBoost Classifier Decision Tree Ensemble (which was ~0.5027 on the upsampled training data). This threshold might need re-evaluation on production data.

The current pipeline expects an Ethnicity column. Do not use it to score real candidates.

## Engineering Integration Recommendations

For integrating this model into a production ATS (like TalentSol), consider these principles and recommended services. These recommendations apply to a retrained model without protected attributes, not to the current saved pipeline.

*   **Start Simple and Iterate:** Start with the current pipeline. Add complexity only when a measured requirement calls for it.
*   **Explainability and Transparency Document:** Document the model, its limitations, the data, and how it produces predictions.
*   **Data Pipeline:** Build a pipeline that applies the saved transformers to new applicant data. Select a deployment service after measuring latency and throughput requirements.
*   **Data and Model Storage:** Securely store raw data, processed data, and the trained model pipeline. Cloud Object Storage (e.g., Cloud Storage, S3, Blob Storage) is suitable for raw data and model files. Managed databases (e.g., Cloud SQL, RDS, Azure SQL) can store structured applicant data. Consider a Feature Store for managing and serving features consistently.
*   **Latency:** Monitor and optimize prediction latency. Ensemble models require running multiple base models, which can increase latency compared to a single model.
*   **Memory:** Be mindful of the memory footprint of the loaded pipeline and its multiple base estimators. HashingVectorizer stores no vocabulary, so it uses little memory. Other parts of the pipeline and the ensembles can still need a lot of memory. High-RAM runtimes are beneficial for training and inference.
*   **Vector Databases:** While the selected XGBoost model utilizes HashingVectorizer, understanding the architectural implications of dense embedding-based models (like Word2Vec or FastText if successfully integrated in the future) is valuable for future iterations. For models utilizing dense vector embeddings, consider the use of a dedicated vector database for efficient storage, indexing, and retrieval at scale.
*   **Scalability:** Measure latency, memory use, and throughput before selecting a deployment platform. This project did not benchmark those properties.
*   **Monitoring:** Implement monitoring for performance tracking (e.g., inference time, error rates), data drift (changes in input data characteristics), and model drift (degradation in model performance over time).
*   **API Endpoint:** Provide a documented API for integration with the ATS.
*   **Security:** Ensure secure handling and storage of sensitive applicant data and the model artifact.
*   **Versioning:** Use a system for model versioning to manage updates and rollbacks effectively.
*   **Computational Resources:** Measure training time, inference time, and memory use before selecting CPU, GPU, TPU, or memory capacity. This project does not report those benchmarks.
*   **Automated Retraining:** Set up automated pipelines for retraining the model periodically with new data to ensure it remains relevant and accurate over time.
*   **Model Interpretation Considerations:** If providing explanations for individual predictions is a requirement, for example, for recruiters, integrate model interpretation techniques, such as SHAP values, alongside the deployed model.
*   **Need for a Human-Centered User Interface:** A human-centered application user interface is still required to serve the models and to offer their intended assistive roles for recruiters or recruitment teams. The model is a tool to assist, not replace, human expertise.
*   **Human Review:** Recruiters must review every candidate. They should not use the model score as the final hiring decision.

## Model Risk: Keyword Matching

This section previously gave applicants tips for raising their score. I replaced it because the tips described how to game the model. The XGBoost model scores resumes through HashingVectorizer features. Resumes that repeat words from the job description can score higher, whether or not the candidate fits the role better. The model also scores the Ethnicity column, which no candidate should be scored on. A recruiter should treat the score as one signal for ordering review. The score should not decide the outcome for a candidate.

## Fairness

**What was tested**

- Recall and precision at tuned thresholds for 16 models, on a 20 percent test split of the Kaggle recruitment dataset. I drew this split after oversampling, so it is not an independent test set.
- Removal of Age, Gender, Race1 and Race2 before training.

**What was not tested**

- Selection rates by group, or adverse impact ratios.
- Error rates or calibration by group.
- Proxy features in resume text or job roles that could stand in for a protected characteristic.
- Performance on real applicant data.

**What a production system would need**

- Retraining without protected attributes, with a test set split off before any oversampling.
- An independent bias audit before any use with real candidates.
- Human review of every model recommendation. The model should help a recruiter order the work. It should not reject a candidate on its own.
- A way for candidates to ask for a human review of their application.

## Limitations

- **Ethnicity input.** The saved model uses Ethnicity. It is not acceptable for real hiring until it is retrained without Ethnicity.
- **Test split drawn after oversampling.** Some test rows duplicate training rows. Cross validation folds share duplicates as well. All reported metrics are likely optimistic and need a rerun with a split made before oversampling.
- **Bootstrap sample.** Oversampling with replacement left 1,740 of the original positive rows out of the training data.
- **Dataset.** The resume text follows a fixed template. I have not confirmed whether the Kaggle dataset is synthetic or which license applies to it. The results may not carry over to real resumes.
- **Model provenance.** I have not confirmed that the committed model file came from the same run as the reported metrics.
- **No group fairness testing.** See [Fairness](#fairness).

Refer to the code notebook for detailed implementation of the preprocessing, modeling, and evaluation steps.

## Corrections

On **November 20, 2026**, I checked the statements below against the notebook outputs, saved model file, and committed training CSV and corrected them. I did not change the model, notebook, or data. The earlier README was wrong or incomplete.

| Item | Previously stated | Corrected to | Cause |
|------|-------------------|--------------|-------|
| Responsible AI statement | "removing sensitive demographic features" | Age, Gender, Race1 and Race2 were removed. Ethnicity is still an input, and retraining without it is planned. | The statement left out that Ethnicity remains in the model. |
| Categorical encoding | "'Ethnicity', 'Job Roles', 'Race1', and 'Race2'" were one hot encoded | Only Ethnicity and Job Roles are encoded | Race1 and Race2 are dropped in notebook cell 12. |
| Race column | Split into Race1 and Race2 "to handle different race categories" | Split, then both columns dropped | The README left out the drop. |
| Class imbalance | Oversampling to prevent bias toward a majority class | Classes were close to balanced (5,150 and 4,850). Oversampling with replacement duplicated some rows and left out 1,740 positive rows. | Did not describe what the code did |
| Privacy | Removing four columns would "ensure data privacy" | The columns are not direct inputs. Resume text can still contain personal information. | Overclaim |
| Test set | "held-out test set" | 20 percent test split drawn after oversampling, not independent | The split contains duplicates of training rows. |
| Selected model | "highly optimal" with "potentially faster and more efficient inference" | Selected model with test split metrics. Inference speed was not measured. | No latency measurement |
| Extra Trees and Random Forest | Training performance "was even higher" | Sentence removed | Random Forest training recall (0.719) was lower than its test recall (0.731). |
| AdaBoost | "very high recall at the default threshold" | Default recall 0.605 | Did not match notebook cell 59 |
| Hybrid Voting | "achieved the target recall" | Recall 0.698, below the target | Did not match notebook cell 59 |
| Word embeddings vs. vectorizers | TF-IDF and FastText "performed comparably" and also "showed poor performance" | TF-IDF tree recall 0.200 and FastText tree recall 0.464, both below target | The paragraph contradicted itself. |
| HashingVectorizer memory | "can be memory-intensive" | Uses little memory because it stores no vocabulary | Contradicted the scikit-learn documentation and the earlier line in this README |
| Resume Optimization Tips | Tips for applicants to raise their score | Replaced with a model risk note | The tips described how to game a model that also scores Ethnicity. |
| Approximate metrics | "~72% recall and ~56% precision" and similar approximations for LightGBM, Gradient Boosting, Averaging, Extra Trees and Random Forest | Exact three decimal test split values from notebook cell 59 | The approximations hid the exact notebook values. |
| Unmeasured claims | Robustness to irrelevant features, lower engineering cost than neural networks, and efficient horizontal scaling | States that the project did not measure these properties | No evaluation or benchmark in the notebook |

The 16 model count, the 10 of 16 target count, the XGBoost threshold of 0.5027, and the recall and precision figures for XGBoost, Hybrid Stacking, Extra Trees, Random Forest, LightGBM, Gradient Boosting and Averaging were rechecked against notebook cells 57 and 59 and are accurate as originally published.
