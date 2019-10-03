import json
import re
import random
import fasttext
import sys
from pathlib import Path
import os


def strip_formatting(string: str):
    """
    Lower the string and separate words from punctuation.
    """
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string


def split_data(reviews_path: str, training_data_path: str, test_data_path: str, percent: float):
    """
    Split the data from reviews_path into training_data_path and test_data_path according to percent value.
    """
    with reviews_path.open() as input_file, \
            training_data_path.open("w") as train_output, test_data_path.open("w") as test_output:
        for line in input_file:
            review_data = json.loads(line)
            rating = review_data['stars']
            text = review_data['text'].replace("\n", " ")
            text = strip_formatting(text)
            fasttext_line = f"__label__{rating} {text}"
            if random.random() <= percent:
                test_output.write(fasttext_line + "\n")
            else:
                train_output.write(fasttext_line + "\n")
    return


def test_file_validity(path: str):
    """
    Check path existence and if it leads to a file.
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            return 1
    return 0


def print_results(n, p, r):
    print("Number of samples\t" + str(n))
    print("Precision@{}\t\t{:.3f}".format(1, p))
    print("Recall@{}\t\t{:.3f}".format(1, r))
    print()


def training_process(training_data_path: str, test_data_path: str, model_path: str):
    """
    Create the model, save it, and display the model evaluation metrics on the test_data.
    """
    model = fasttext.train_supervised(input=str(training_data_path), lr=1.0, epoch=25, wordNgrams=2,
                                      loss='hs', bucket=200000, dim=50)
    model.save_model(model_path)
    model = fasttext.load_model(model_path)
    print("--- Model predictions on test_data ---")
    print_results(*(model.test(str(test_data_path))))
    return


def predict_rating(review: str, model_path: str):
    """
    Predict the rating of the review and print it.
    """
    # Pre-process the review so it matches the training format
    preprocessed_review = strip_formatting(review)

    # Load the model
    classifier = fasttext.load_model(model_path)

    # Get fasttext to classify each review with the model
    label, probability = classifier.predict(preprocessed_review, 1)

    # Print the results
    stars = int(label[0][9:-2])
    print("{} ({}% confidence)".format("✰" * stars, int(probability * 100)))
    print(f'"{review}"')
    print()
    return


def main():
    # Set paths
    dataset_folder = Path("../dataset")
    reviews_path = dataset_folder / "review.json"
    training_data_path = dataset_folder / "fasttext_dataset_training.txt"
    test_data_path = dataset_folder / "fasttext_dataset_test.txt"
    model_path = "../models/reviews_model_ngrams2.bin"
    percent_test_data = 0.10

    # Split dataset into test and training set according to percent_test_data
    if test_file_validity(training_data_path) == 0 or test_file_validity(test_data_path) == 0:
        print("Starting splitting ...")
        split_data(reviews_path, training_data_path, test_data_path, percent_test_data)
        print("... splitting done.")

    # Create model and display model evaluation metrics on test_data
    if test_file_validity(model_path) == 0:
        training_process(training_data_path, test_data_path, model_path)
    # Predict the rating of the review
    elif len(sys.argv) == 2:
        predict_rating(sys.argv[1], model_path)
    return


if __name__ == "__main__":
    main()
