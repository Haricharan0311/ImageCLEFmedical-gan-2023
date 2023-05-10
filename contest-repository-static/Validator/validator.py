import csv
import sys
import os


class ImageCLEFGansTask1Validator:
    def __init__(self, image_ids_file_path):
        self.image_ids_file_path = image_ids_file_path
        self._load_image_ids()

    def _load_image_ids(self):
        with open(self.image_ids_file_path) as file:
            self.image_ids = [line.rstrip() for line in file]

    def validate_format(self, submission_file_path):
        print("validating format...")

        with open(submission_file_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=',',
                                quoting=csv.QUOTE_NONE)
            line_nbr = 0
            predictions = {}

            for row in reader:
                line_nbr += 1

                # Not 2 comma separated tokens on line => Error
                if len(row) != 2:
                    self._raise_exception("Wrong format: Each line must consist of 2 tokens separated by a comma. You have to specify a figure ID "
                                          + "followed by a score of 0 or 1 to indicates whether the image was used for training or not (0=not used, 1=used). "
                                            + "Each token must be separated by a comma ({}).", line_nbr, "<figure_id>, <0 | 1>")

                image_id = row[0].strip()

                # Figure ID not contained in testset => Error
                if image_id not in self.image_ids:
                    self._raise_exception(
                        "Figure ID '{}' is not contained in testset.", line_nbr, image_id)

                # Figure ID contained more then once => Error
                if image_id in predictions:
                    self._raise_exception(
                        "Figure ID '{}' was specified more than once in submission file.", line_nbr, image_id)

                score = row[1].strip()

                # Score not 0 or 1 => Error
                try:
                    score = int(score)
                    if score != 0 and score != 1:
                        raise ValueError
                except ValueError:
                    self._raise_exception(
                        "Wrong score '{}'. Score must be either 0 or 1 (0=not used for training, 1=used for training).", line_nbr, score)

                predictions[image_id] = score

            # Not all figure IDs included => Error
            if len(predictions) != len(self.image_ids):
                self._raise_exception(
                    "Number of figure IDs in submission file not equal to number of figure IDs in testset.", line_nbr)

            print("The format of your submission file '{}' is correct. Good Job!".format(
                submission_file_path))

    def _raise_exception(self, message, line_nbr, *args):
        raise SubmissionValidationException(message.format(*args) +
                                            " Error occured at line nbr {}.".format(line_nbr))


class SubmissionValidationException(Exception):
    pass


if __name__ == "__main__":

    usage_message = "Usage: validator.py submission_file_path"

    if len(sys.argv) != 2 or (len(sys.argv) == 2 and str(sys.argv[1]) == '--help'):
        print(usage_message)
        sys.exit()

    submission_file_path = str(sys.argv[1])

    if not os.path.isfile(submission_file_path):
        print("{} is not an existing file".format(submission_file_path))
        sys.exit()

    image_ids_file_path = "./resources/image_ids.txt"

    imageCLEFGansTask1Validator = ImageCLEFGansTask1Validator(
        image_ids_file_path)

    try:
        imageCLEFGansTask1Validator.validate_format(submission_file_path)
    except SubmissionValidationException as ex:
        print(ex)
