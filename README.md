# TrainedCSVforCoreML
Key Steps:
Read CSV File:

Handles potential encoding issues (utf-8 or latin-1) to ensure compatibility with different datasets.

Inspect Data:

Examines the structure and content of the dataset.

Validates that the required columns exist and checks for missing labels.

Clean Labels:

Maps numeric sentiment labels (e.g., 0 -> negative, 4 -> positive) to descriptive categories.

Ensures at least two distinct sentiment classes are present for meaningful analysis.

Text Preprocessing:

Cleans the text column by:

Removing URLs.

Eliminating special characters and digits.

Converting all text to lowercase for uniformity.

Save Cleaned Data:

Outputs the cleaned data to a new CSV file (cleaned_file.csv).

Split Dataset:

Divides the cleaned dataset into:

Training Set: Used to train the sentiment analysis model.

Validation Set: Used to test the model's performance.

Saves these subsets as CSV files (train_data.csv and validation_data.csv).

Analyze Label Distribution:

Checks the distribution of sentiment labels in both training and validation sets.

Outputs:
Cleaned Dataset: cleaned_file.csv – preprocessed data ready for training.

Training Set: train_data.csv – used for model training.

Validation Set: validation_data.csv – used for model validation.
