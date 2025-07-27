# TalentSol - Applicant Tracking System Learning Project (Machine Learning Component)

This repository contains the machine learning component of the TalentSol project, focusing on training supervised classifiers to predict and recommend candidates for prioritization for a given job role and description. This is a learning project exploring different text embedding techniques and ensemble methods for a real-world recruitment use case.

## Project Goal

The primary goal of this project is to train a supervised classifier that can predict which job applicants are the "Best Match" for a given job role and description, thereby helping recruiters prioritize candidates during the screening process. A key focus is achieving high recall to minimize false negatives (missing out on potentially good candidates) while maintaining reasonable precision.

## Dataset

The project uses the `job_applicant_dataset.csv` dataset, which contains information about job applicants, their resumes, and whether they were considered a "Best Match" for a specific job role and description.

**Dataset Source:** The dataset is sourced from Kaggle: [Recruitment Dataset](https://www.kaggle.com/datasets/surendra365/recruitement-dataset)

## Preprocessing and Feature Engineering

Before we can train a machine learning model on text data like job descriptions and resumes, we need to convert the words and phrases into a numerical format that the model can understand. This process is called **Feature Engineering**, where we create meaningful numerical features from the raw data.

The following preprocessing and feature engineering steps were performed on the dataset:

1.  **Data Loading:** The `job_applicant_dataset.csv` file was loaded into a pandas DataFrame.
2.  **Identifier and Sensitive Feature Removal:** Columns like 'Job Applicant Name', 'Job Applicant ID', 'Age', and 'Gender' were removed early in the preprocessing pipeline. These columns contain **personally identifiable information (PII)** and were removed to mitigate risks of re-identification and ensure data privacy, aligning with responsible AI considerations. They were also not considered relevant or appropriate for the prioritization task in this version of the project, as the focus is on text matching between the resume/CV and the job description/job role.
3.  **Categorical Feature Splitting:** The 'Race' column, containing multiple race values, was split into separate columns ('Race1' and 'Race2') to handle different race categories.
4.  **Missing Value Handling:** Checked for and confirmed no missing data points were present after initial cleaning.
5.  **Handling Class Imbalance:** The dataset had more candidates who were *not* a "Best Match" than those who were. To prevent the model from being biased towards the majority class, we used a technique called **oversampling** to create more examples of the minority class ("Best Match").
6.  **Categorical Encoding:** Non-numerical categories like 'Ethnicity', 'Job Roles', 'Race1', and 'Race2' were converted into numerical representations using One-Hot Encoding.
7.  **Text Feature Embedding:** This is where we convert the text from 'Job Description' and 'Resume' into numbers. We explored different methods:
    *   **CountVectorizer:** Creates a vocabulary of all unique words and counts their occurrences in each document.
    *   **TF-IDF Vectorization (Term Frequency-Inverse Document Frequency):** Calculates a score for each word based on its frequency within a document and its rarity across all documents.
    *   **HashingVectorizer:** A memory-efficient vectorizer that maps text features to a fixed-size vector using hashing.
    *   **Word2Vec Embeddings:** Learns dense numerical vector representations for words, capturing semantic relationships. Document vectors were created by averaging word vectors.
    *   **FastText Embeddings:** An extension of Word2Vec that considers sub-word information, useful for out-of-vocabulary words. Document vectors were created by averaging word vectors.

## Modeling and Evaluation

Various Decision Tree-based models and ensemble methods were trained and evaluated using **5-fold Stratified Cross-Validation** on the upsampled training data. This cross-validation approach ensures that each fold has a representative proportion of both classes ('Best Match' and 'Not Best Match').

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

The models were evaluated based on standard metrics (Accuracy, Precision, Recall, F1-score, ROC AUC) on a held-out test set. Crucially, **Precision-Recall curves** were analyzed for each model's predicted probabilities (using Out-of-Fold predictions on the training data), and **decision thresholds were optimized to achieve a target recall level** (approximately 70%). This optimization aligns the model's decision-making with the project's goal of prioritizing candidates and minimizing false negatives. Confusion matrices were visualized at these optimized thresholds to understand the specific performance trade-offs.

**Methodology for Prediction (How the Model Makes a Prediction):**

For a new applicant, the prediction process involves: applying the *same* preprocessing and feature engineering steps (using the fitted transformers from the chosen trained pipeline), scaling the features, feeding the scaled features to the trained classifier to get a probability score, and finally applying the **optimized decision threshold** (derived during evaluation) to classify the applicant as 'Best Match' (1) or 'Not Best Match' (0). For ensemble models, this process involves obtaining probabilities from the base models and combining them according to the ensemble strategy before applying the optimized threshold.

**Responsible AI Considerations:**

Responsible AI considerations were incorporated by removing sensitive demographic features and using evaluation metrics (like recall and confusion matrices at optimized thresholds) that are relevant to the task's goals while acknowledging potential biases. Further steps for a production system would include bias audits, fairness metrics, human oversight, and transparency.

## Results

The models were evaluated at decision thresholds optimized to achieve approximately 70% recall based on the upsampled training data. A comparison was then made on the held-out test set using both default (0.5) and optimized thresholds. Key findings include:

*   **Impact of Threshold Optimization:** Optimizing the decision threshold significantly impacted the trade-off between Precision and Recall for most models compared to using the default 0.5 threshold. Models were able to achieve higher recall, often at the cost of reduced precision, as clearly shown in the performance comparison plots in the notebook.
*   **Target Recall Achievement:** 10 out of the 16 evaluated models successfully achieved or exceeded the target recall of 70% on the held-out test set when using their optimized thresholds. This demonstrates the potential for multiple models to effectively identify a high percentage of "Best Match" candidates.
*   **Most Optimal Solution:** The **XGBoost Classifier Ensemble** emerged as a highly optimal solution after the final evaluation at optimized decision thresholds. This model achieved the target recall of ~70% with high precision (~57%) on the held-out test set, matching the highest precision among models that met the recall target while offering potentially faster and more efficient inference compared to the Hybrid Stacking Ensemble.
*   **Hybrid Stacking Ensemble Performance:** The **Hybrid Stacking Ensemble** also performed very well, achieving the target recall of ~70% with the highest precision (~57%) on the held-out test set. While complex, it demonstrates the power of combining multiple model perspectives.
*   **HashingVectorizer Ensemble Performance:** The **Extra Trees Decision Tree Ensemble** and **Random Forest Decision Tree Ensemble**, both utilizing HashingVectorizer for text preprocessing, showed strong performance. At their optimized thresholds on the *held-out test set*, Extra Trees achieved ~73% recall and ~53% precision, and Random Forest achieved ~73% recall and ~53% precision. Their performance on the upsampled training data was even higher, indicating potential for very high recall or precision if the test set distribution was closer to the upsampled training data. They offer the highest achieved recall among the top performers.
*   **Gradient Boosting Models:** LightGBM (achieving ~72% recall and ~56% precision) also performed well on the test set at its optimized threshold, exceeding the target recall with competitive precision. Gradient Boosting (~71% recall and ~54% precision) also met the target.
*   **AdaBoost Performance:** AdaBoost achieved very high recall at the default threshold but with significantly lower precision. At its optimized threshold for 70% recall on the training data, its performance on the test set was just below the target (~69.9% recall, ~55% precision), suggesting it might not generalize as well as other boosting methods or that its optimization was less effective on the test distribution.
*   **Averaging vs. Voting/Stacking:** The simple Averaging ensemble (~69% recall, ~55% precision) performed reasonably well, just slightly missing the 70% recall target on the test set. The Hybrid Voting (~70% recall, ~55% precision) ensemble achieved the target recall on the test set with good precision.
*   **Word Embeddings vs. Vectorizers:** Decision Trees with traditional vectorizers (CountVectorizer, TF-IDF, HashingVectorizer) and FastText embeddings performed comparably at the 70% recall target on the test set, with HashingVectorizer ensembles performing slightly better. The TF-IDF and FastText individual Decision Trees showed poor performance at their optimized thresholds on the test set. Word2Vec also met the target recall with reasonable precision.
*   **Best Performers for Prioritization:** Considering the goal of prioritizing candidates, high recall, while maintaining reasonable precision on the held-out test set, the **XGBoost Classifier Decision Tree Ensemble** and the **Hybrid Stacking Ensemble** achieved the 70% recall target with the highest precision, ~57%. The **HashingVectorizer Decision Tree Ensemble** (both Extra Trees and Random Forest) also met the target recall (~73%) with slightly lower precision (~53%), representing a different point on the precision-recall trade-off curve that might be acceptable depending on the specific business needs. The **Gradient Boosting Classifier** and **LightGBM Classifier** also met the target recall with competitive precision.
*   **Multi-view Learning:** Multi-view Learning was explored theoretically, identifying text and categorical data as potential views and discussing fusion strategies, benefits, and challenges, concluding that a full implementation was outside the current scope but laid the theoretical foundation.

### Conceptual Strengths of Decision Tree Models

Decision Tree models and their ensembles are well-suited for this text matching task due to several conceptual strengths:

*   **Handling Non-linear Relationships and Feature Interactions:** They naturally capture complex, non-linear relationships and interactions between features without assuming linear boundaries.
*   **Tolerance to Irrelevant Features:** They are relatively robust to including irrelevant or noisy features.
*   **Interpretability (for individual trees):** Individual trees are highly interpretable, providing rule sets for predictions.
*   **No Assumptions about Data Distribution:** They are non-parametric, suitable for complex feature distributions like text and categorical data.
*   **Engineering Complexity and Cost vs. Neural Networks:** Compared to large neural networks (BERT, LLMs), Decision Tree models generally have significantly lower engineering complexity, computational costs for training and inference, and smaller memory footprints, making them more practical for deployment in resource-constrained environments like an ATS.

## Decision Trees and Data Samples

Decision Trees require numerical input. Text data is converted via vectorization/embedding (CountVectorizer, TF-IDF, HashingVectorizer, Word2Vec, FastText) and categorical data via encoding (One-Hot Encoding). Trees work by recursively splitting the data based on features that best reduce impurity (e.g., using Gini Impurity or Entropy).

## Why Some Models Fail to Meet the Target Recall

Even with optimized thresholds, some models may not reach the target recall on the held-out test set due to:

*   **Differences in Data Distribution:** The test set may have a slightly different distribution than the training data used for optimization.
*   **Model Limitations:** Some models may intrinsically be unable to achieve high recall with reasonable precision on this dataset.
*   **Overfitting:** While mitigated by OOF predictions, slight overfitting to training splits can occur.
*   **Inherent Problem Difficulty:** The task of matching resumes to job descriptions is complex, potentially limiting achievable performance regardless of the model.

## Considerations for Alternative Pruning Techniques

Standard Decision Trees use pre-pruning (e.g., `max_depth`, `min_samples_split`, `min_samples_leaf`). Post-pruning techniques like Cost-Complexity Pruning (`ccp_alpha`) are also available and can be tuned via GridSearchCV. Ensemble methods primarily regularize through aggregation, but tuning base learner complexity remains important.

## Saved Artifacts

The following artifacts from the modeling process have been saved:

*   `upsampled_training_data.csv`: The final upsampled dataset used for training.
*   `best_performing_model_pipeline.joblib`: The trained scikit-learn Pipeline object for the **XGBoost Classifier Decision Tree Ensemble**. This is the recommended artifact for making predictions on new data.

## Inference and Integration

To use the trained model for predicting on new job applicants and integrating into a system:

1.  **Load the Trained Pipeline:** Load the `best_performing_model_pipeline.joblib` file using `joblib.load()`.
2.  **Predict:** Apply the loaded pipeline's `predict_proba()` method to new, unseen applicant data (after ensuring it has the expected column structure). The output probabilities can then be thresholded using the optimized decision threshold found for the XGBoost Classifier Decision Tree Ensemble (which was ~0.5027 on the upsampled training data). This threshold might need re-evaluation on production data.

## Engineering Integration Recommendations

For integrating this model into a production ATS (like TalentSol), consider these principles and recommended services:

*   **Start Simple and Iterate:** Begin with a straightforward and lightweight architecture. This is a **key engineering tenet**. For a problem like initial candidate screening, which involves handling a high volume of candidates and prioritizing for high recall, starting simple allows for faster experimentation and testing to quickly determine if the solution is effective before adding complexity. Avoid unnecessary complexity ("bloat") that could negatively impact inference performance or operational costs. Iterate and add complexity only when necessary to meet specific requirements.
*   **Explainability and Transparency Document:** Create a clear and easy-to-understand document explaining how the model works, its limitations, the data used, and how predictions are made. This document is crucial for users of the system (recruiters, HR personnel) and other stakeholders (business leads, legal) to build trust, ensure fair usage, and understand the model's impact on the hiring process.
*   **Real-time Data Pipeline:** Implement an efficient pipeline to preprocess new applicant data using the loaded pipeline's transformers. Recommended services for building data pipelines include serverless functions (e.g., Cloud Functions, Lambda), containerization platforms (e.g., Cloud Run, Kubernetes), or batch processing services (e.g., Dataflow, EMR) depending on latency and throughput requirements.
*   **Data and Model Storage:** Securely store raw data, processed data, and the trained model pipeline. Cloud Object Storage (e.g., Cloud Storage, S3, Blob Storage) is suitable for raw data and model files. Managed databases (e.g., Cloud SQL, RDS, Azure SQL) can store structured applicant data. Consider a Feature Store for managing and serving features consistently.
*   **Latency:** Monitor and optimize prediction latency. Ensemble models require running multiple base models, which can increase latency compared to a single model.
*   **Memory:** Be mindful of the memory footprint of the loaded pipeline and its multiple base estimators. HashingVectorizer used by some base estimators can be memory-intensive. High-RAM runtimes are beneficial for training and inference.
*   **Vector Databases:** While the selected XGBoost model utilizes HashingVectorizer, understanding the architectural implications of dense embedding-based models (like Word2Vec or FastText if successfully integrated in the future) is valuable for future iterations. For models utilizing dense vector embeddings, consider the use of a dedicated vector database for efficient storage, indexing, and retrieval at scale.
*   **Scalability:** Deploy the model as a scalable service using platforms like Kubernetes or serverless functions to handle varying request loads. The generally efficient inference of tree-based models facilitates horizontal scaling.
*   **Monitoring:** Implement monitoring for performance tracking (e.g., inference time, error rates), data drift (changes in input data characteristics), and model drift (degradation in model performance over time).
*   **API Endpoint:** Provide a clear and well-documented API for seamless integration with other components of the ATS.
*   **Security:** Ensure secure handling and storage of sensitive applicant data and the model artifact.
*   **Versioning:** Use a system for model versioning to manage updates and rollbacks effectively.
*   **Computational Resources:** Using powerful hardware accelerators (like GPUs or TPUs if applicable for future models, or simply high-CPU/high-RAM VMs for current models) can significantly speed up training and inference, especially with large datasets and complex ensembles. The high RAM requirements observed during this project underscore the need for sufficient memory resources in the deployment environment.
*   **Automated Retraining:** Set up automated pipelines for retraining the model periodically with new data to ensure it remains relevant and accurate over time.
*   **Model Interpretation Considerations:** If providing explanations for individual predictions is a requirement, for example, for recruiters, integrate model interpretation techniques, such as SHAP values, alongside the deployed model.
*   **Need for a Human-Centered User Interface:** A human-centered application user interface is still required to serve the models and to offer their intended assistive roles for recruiters or recruitment teams. The model is a tool to assist, not replace, human expertise.
*   **Sub-symbolic Human Intuition is Still Crucial:** The models can play an important assistive role to save time and undifferentiated heavy lifting, but users (recruiters) should also rely on a variety of other tools (BI tools, spreadsheets, manual review) and their own expertise and intuition, especially for candidates not classified as "Best Match" or for identifying exceptional talent from unexpected backgrounds.

## Resume Optimization Tips (Based on XGBoost Classifier Decision Tree Ensemble)

Based on the selection of the XGBoost Classifier Decision Tree Ensemble as a highly optimal performer, here are some generalized tips for job applicants to optimize their resumes/CVs:

*   **Relevant Keyword Usage:** Ensure your resume includes keywords and phrases directly relevant to the job description and industry. XGBoost, using HashingVectorizer, will capture the presence and frequency of these terms.
*   **Clear and Structured Content:** Organize your resume logically with clear headings. This helps the preprocessing steps accurately extract and vectorize the text content.
*   **Tailor to the Job:** While XGBoost is robust, tailoring your resume to the specific job description by incorporating relevant terminology will likely improve your match score.
*   **Highlight Key Skills and Experience:** Clearly articulate your skills and experience that align with the job requirements.

The XGBoost model's performance relies on the numerical features derived from the text and categorical data. A well-structured resume with relevant keywords and tailored content will provide better input for the HashingVectorizer and categorical encoding, leading to a more accurate prediction.

Refer to the code notebook for detailed implementation of the preprocessing, modeling, and evaluation steps.
