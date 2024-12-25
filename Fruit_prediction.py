import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree
import matplotlib.pyplot as plt

# Load the dataset
file_path = "Fruits.csv"
data = pd.read_csv(file_path)

# Drop unnecessary columns
data = data.drop(columns=['Unnamed: 0', 'fruit_subtype'])

# Separate features and target
X = data[['mass', 'width', 'height', 'color_score']]
y = data['fruit_name']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier with class weight balancing
clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# Train the model
clf.fit(X_train, y_train)

# Perform cross-validation
cv_scores = cross_val_score(clf, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean()}")

# Save the trained model and metadata to a file
model_file = "fruit_prediction_model.joblib"
model_data = {
    'model': clf,
    'accuracy': cv_scores.mean(),
    'features': ['mass', 'width', 'height', 'color_score'],
    'classes': list(clf.classes_)
}
joblib.dump(model_data, model_file)
print(f"Model saved to {model_file}")

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
tree.plot_tree(
    clf,
    feature_names=['mass', 'width', 'height', 'color_score'],
    class_names=clf.classes_,
    filled=True
)
plt.title("Decision Tree Visualization")
plt.savefig("decision_tree.png")
plt.close()

# User testing functionality
def test_model():
    # Load the saved model and metadata
    model_data = joblib.load(model_file)
    clf = model_data['model']
    print("Model loaded for testing.")

    # Get user input for prediction
    def get_float_input(prompt):
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    print("Enter the features for a fruit:")
    mass = get_float_input("Mass: ")
    width = get_float_input("Width: ")
    height = get_float_input("Height: ")
    color_score = get_float_input("Color Score: ")

    # Prepare the input as a DataFrame
    user_input = pd.DataFrame([[mass, width, height, color_score]], columns=['mass', 'width', 'height', 'color_score'])

    # Make a prediction
    prediction = clf.predict(user_input)
    print(f"Predicted fruit label: {prediction[0]}")

test_model()
