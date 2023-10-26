# Import necessary libraries and modules
from metaflow import FlowSpec, step
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Define a Metaflow flow class named JenkinsLogFlow
class JenkinsLogFlow(FlowSpec):

    # Step 1: Reading and combining failure and success logs
    @step
    def start(self):
        # Read failure logs from a file
        with open("failure_logs.txt", "r") as f:
            self.failure_logs = [line.strip() for line in f.readlines()]

        # Read success logs from a file
        with open("success_logs.txt", "r") as f:
            self.success_logs = [line.strip() for line in f.readlines()]

        # Combine logs and labels, assign labels (1 for failure, 0 for success)
        self.logs = self.failure_logs + self.success_logs
        self.labels = [1] * len(self.failure_logs) + [0] * len(self.success_logs)

        # Move to the next step for data preparation
        self.next(self.prepare_data)

    # Step 2: Data preparation - converting text logs to numerical features
    @step
    def prepare_data(self):
        # Convert text logs to numerical features using TF-IDF vectorization
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(self.logs)
        y = self.labels

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

        # Move to the next step for model training
        self.next(self.train_model)

    # Step 3: Model training - training a Random Forest classifier
    @step
    def train_model(self):
        # Train a Random Forest classifier with 100 estimators
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(self.X_train, self.y_train)

        # Move to the next step for model validation
        self.next(self.validate_model)

    # Step 4: Model validation - evaluating the trained model
    @step
    def validate_model(self):
        # Validate the trained model on the test data
        predictions = self.model.predict(self.X_test)
        
        # Calculate and store model accuracy
        self.accuracy = accuracy_score(self.y_test, predictions)
        
        # Generate and store a classification report
        self.classification_rep = classification_report(self.y_test, predictions)

        # Print model accuracy and classification report
        print(f"Model accuracy: {self.accuracy:.2f}")
        print("Classification Report:")
        print(self.classification_rep)

        # Move to the next step for saving the model
        self.next(self.save_model)

    # Step 5: Saving the trained model and vectorizer to disk
    @step
    def save_model(self):
        # Save the trained model and TF-IDF vectorizer to disk as pickle files
        joblib.dump(self.model, 'trained_model.pkl')
        joblib.dump(self.vectorizer, 'tfidf_vectorizer.pkl')

        # Move to the final step to mark the completion of the flow
        self.next(self.end)

    # Step 6: Final step - Flow completion
    @step
    def end(self):
        print("JenkinsLogFlow is completed.")

# Entry point: If the script is executed directly, create and run the JenkinsLogFlow
if __name__ == "__main__":
    JenkinsLogFlow()
