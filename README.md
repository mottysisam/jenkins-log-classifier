# JenkinsLogFlow Project Documentation

## Repository Name
**jenkins-log-classifier**

## Description
A machine learning pipeline designed to classify Jenkins logs into `FAILURE` or `SUCCESS` categories. The project uses the Metaflow framework to structure the pipeline and a Random Forest classifier from scikit-learn for the actual classification task.

---

## Overview
The `jenkins-log-classifier` project processes Jenkins logs to identify whether they indicate successful operations or failures. This classification can assist DevOps and MLOps teams in quickly diagnosing and addressing issues in their CI/CD pipelines.

---

## Workflow Steps

1. **Start**: Initializes the dataset. This step sets up mock Jenkins logs for demonstration purposes. In a real-world scenario, this data would be collected and labeled accordingly.
2. **Prepare Data**: This step preprocesses the Jenkins logs. It uses a TF-IDF Vectorizer to convert the text logs into a numerical format suitable for machine learning models.
3. **Train Model**: A Random Forest classifier is trained on the preprocessed logs to distinguish between `FAILURE` and `SUCCESS`.
4. **Validate Model**: The trained model's performance is evaluated on a test set. Accuracy and a detailed classification report (including precision, recall, and F1-score) are printed.
5. **Save Model**: The trained Random Forest model and the TF-IDF vectorizer are saved to disk as `.pkl` files. This ensures that they can be loaded and used without retraining or refitting.
6. **End**: Signals the completion of the flow.

---

## How to Run
Ensure that all dependencies, such as Metaflow and scikit-learn, are installed. Then, navigate to the directory containing the script and run:

```python3.10 jtraining2.py run```

After execution, you will find two files in the directory: trained_model.pkl and tfidf_vectorizer.pkl, representing the trained model and the vectorizer, respectively.

---


## Future Enhancements

- **Data Collection**: Implement mechanisms to collect and label a substantial amount of real Jenkins logs.
- **Hyperparameter Tuning**: Use techniques like grid search or random search to optimize the Random Forest classifier's parameters.
- **Model Selection**: Experiment with other classifiers or deep learning models for potentially better performance.
- **Deployment**: Set up an API or a web interface to classify real-time Jenkins logs using the trained model.

---

## Contribution
Contributors are welcome! Please fork the repository and submit pull requests for any enhancements, fixes, or features you'd like to add.

---
