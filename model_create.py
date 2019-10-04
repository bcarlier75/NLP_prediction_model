import json
import random
import fasttext
import model_utils


def split_data(reviews_path, training_data_path, test_data_path, percent: float):
    """Split the data from reviews_path into training_data_path and test_data_path according to percent value."""
    with reviews_path.open() as input_file, \
            training_data_path.open("w") as train_output, test_data_path.open("w") as test_output:
        for line in input_file:
            review_data = json.loads(line)
            rating = review_data['stars']
            text = review_data['text'].replace("\n", " ")
            text = model_utils.strip_formatting(text)
            fasttext_line = f"__label__{rating} {text}"
            if random.random() <= percent:
                test_output.write(fasttext_line + "\n")
            else:
                train_output.write(fasttext_line + "\n")
    return


def process_training(training_data_path: str, test_data_path: str, model_path: str):
    """Create the model, save it, and display the model evaluation metrics on the test_data."""
    print("--- Creating model, this may take a while ---")
    # Model hyperparameters can be tuned here. See https://fasttext.cc/docs/en/options.html
    model = fasttext.train_supervised(input=str(training_data_path),
                                      lr=1.0, epoch=25,
                                      wordNgrams=2, loss='hs',
                                      bucket=200000, dim=50)
    model.save_model(model_path)
    print("--- Done ---")
    print("--- Model prediction metrics on test data ---")
    model_utils.print_results(*(model.test(str(test_data_path))))
    return
