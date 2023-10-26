# Import necessary libraries
from metaflow import FlowSpec, step
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Define the main flow for processing and classifying Jenkins logs
class JenkinsLogFlow(FlowSpec):

    @step
    def start(self):
        # Initialize dataset with mock Jenkins logs
        # This is a sample dataset for demonstration purposes
        self.logs = [
            "Build failed due to timeout.",
            "Compilation error in module X.",
            "Tests passed successfully.",
            "Deployment completed without errors.",
            "Error connecting to database.",
            "Successfully fetched the latest code.",
            "Memory leak detected in module Y.",
            "Connection timeout while fetching dependencies.",
            "All unit tests passed.",
            "Successfully built the Docker image."
        ]
        # Labels: 1 represents FAILURE and 0 represents SUCCESS
        self.labels = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]

        # Proceed to the next step: Data Preparation
        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        # Convert text logs into a numerical matrix using TF-IDF vectorization
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(self.logs)
        y = self.labels

        # Split the dataset into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

        # Proceed to the next step: Model Training
        self.next(self.train_model)

    @step
    def train_model(self):
        # Initialize and train a Random Forest classifier on the training data
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(self.X_train, self.y_train)

        # Proceed to the next step: Model Validation
        self.next(self.validate_model)

    @step
    def validate_model(self):
        # Predict the labels for the test set
        predictions = self.model.predict(self.X_test)

        # Calculate the accuracy and other metrics for the predictions
        self.accuracy = accuracy_score(self.y_test, predictions)
        self.classification_rep = classification_report(self.y_test, predictions)

        # Print the results
        print(f"Model accuracy: {self.accuracy:.2f}")
        print("Classification Report:")
        print(self.classification_rep)

        # Proceed to the next step: Model Saving
        self.next(self.save_model)

    @step
    def save_model(self):
        # Serialize and save the trained model and vectorizer to disk
        joblib.dump(self.model, 'trained_model.pkl')
        joblib.dump(self.vectorizer, 'tfidf_vectorizer.pkl')

        # Proceed to the final step
        self.next(self.end)

    @step
    def end(self):
        # Signal the completion of the flow
        print("JenkinsLogFlow is completed.")

# Execute the flow if the script is run as the main module
if __name__ == "__main__":
    JenkinsLogFlow()
