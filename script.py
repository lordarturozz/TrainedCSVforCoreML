import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Read the CSV file with error handling for encoding
file_path = 'traning.csv'
try:
    # Try reading with utf-8 encoding
    df = pd.read_csv(file_path, encoding='utf-8', header=None)
except UnicodeDecodeError:
    try:
        # Fallback to latin-1 encoding if utf-8 fails
        df = pd.read_csv(file_path, encoding='latin-1', header=None)
    except Exception as e:
        print(f"Error reading file: {e}")
        exit()

# Inspect the DataFrame
print("First 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns)

# Ensure the file has enough columns
if len(df.columns) > 5:
    df = df[[0, 5]]  # Access columns by integer indices
    df.columns = ['label', 'text']  # Rename the columns
else:
    print("The file does not have enough columns.")
    exit()

# Check for NaN values in the label column
print("Number of NaN values in 'label' column:", df['label'].isna().sum())

# Drop rows with NaN labels
df = df.dropna(subset=['label'])

# Check unique values in the label column
print("Unique values in 'label' column before mapping:", df['label'].unique())

# Update label mapping based on your dataset
label_mapping = {0: "negative", 4: "positive"}  # Only negative and positive
df["label"] = df["label"].map(label_mapping)

# Check unique values in the label column after mapping
print("Unique values in 'label' column after mapping:", df['label'].unique())

# Check label distribution
print("Label distribution:")
print(df['label'].value_counts())

# Exit if there is only one class
if len(df['label'].unique()) < 2:
    print("Error: The dataset contains only one class. At least two classes are required.")
    exit()

# Clean the text column
def clean_text(text):
    if not isinstance(text, str):  # Handle non-string values
        return ""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

df["text"] = df["text"].apply(clean_text)

# Save the cleaned data
df.to_csv('cleaned_file.csv', index=False, encoding='utf-8')
print("File cleaned and saved as 'cleaned_file.csv'")

# Split the data into training and validation sets for Create ML
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# Save the training and validation data for Create ML
train_df.to_csv('train_data.csv', index=False, encoding='utf-8')
val_df.to_csv('validation_data.csv', index=False, encoding='utf-8')
print("Training and validation data saved for Create ML.")

# Check label distribution in the training and validation sets
print("Label distribution in training set:")
print(train_df['label'].value_counts())
print("Label distribution in validation set:")
print(val_df['label'].value_counts())