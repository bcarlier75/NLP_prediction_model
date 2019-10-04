import fasttext
import sys
from pathlib import Path
import model_create
import model_utils


def predict_rating(review: str, model_path: str):
    """Predict the rating of the review and print it."""
    # Pre-process the review so it matches the training format
    preprocessed_review = model_utils.strip_formatting(review)

    # Load the model
    classifier = fasttext.load_model(model_path)

    # Get fasttext to classify each review with the model
    label, probability = classifier.predict(preprocessed_review, 1)

    # Print the results
    stars = int(label[0][9:-2])
    print("{} ({}% confidence)".format("✰" * stars, int(probability * 100)))
    print(f"{review}")
    print()
    return


def main():
    # Set paths and variables
    dataset_folder = Path("../dataset")
    reviews_path = dataset_folder / "review.json"
    training_data_path = dataset_folder / "fasttext_dataset_training.txt"
    test_data_path = dataset_folder / "fasttext_dataset_test.txt"
    model_path = "../models/reviews_model_ngrams2.bin"
    percent_test_data = 0.10

    # Split dataset into test and training set according to percent_test_data
    if model_utils.test_file_validity(model_path) == 0:
        if model_utils.test_file_validity(training_data_path) == 0 \
                or model_utils.test_file_validity(test_data_path) == 0:
            print("--- Start the splitting of the dataset, this may take a while ---")
            model_create.split_data(reviews_path, training_data_path, test_data_path, percent_test_data)
            print("--- Done ---")
        # Create model and display model evaluation metrics on test_data
        model_create.process_training(training_data_path, test_data_path, model_path)
        # Predict the rating of the review
    if len(sys.argv) == 2:
        predict_rating(sys.argv[1], model_path)
    else:
        print('Usage: python predit_review_rating.py "My review on a restaurant, bar, hotel ..."')
    return


if __name__ == "__main__":
    main()
