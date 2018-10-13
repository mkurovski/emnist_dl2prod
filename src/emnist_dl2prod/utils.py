"""
Utility functions for data loading, downloading, display
"""
__author__ = "Marcel Kurovski"
__copyright__ = "Marcel Kurovski"
__license__ = "mit"

import json
import logging
import os
import requests
import shutil
import sys
import zipfile

import matplotlib.pyplot as plt

import numpy as np
from scipy.io import loadmat


EMNIST_MATLAB_URL = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'
EMNIST_FILENAME = 'emnist-byclass.mat'


_logger = logging.getLogger(__name__)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


setup_logging(logging.INFO)


def load_emnist(emnist_folder_path, download=False):
    """
    Loads EMNIST data from folder (eventually downloads first)
    using the Matlab format
    See https://www.nist.gov/itl/iad/image-group/emnist-dataset

    Args:
        emnist_folder_path (str): folder that contains EMNIST Matlab files
        download (bool): in case `emnist_folder_path` does not exist or contain
                         the Matlab files, download the data and extract them

    Returns:
        x_train (:obj:`np.array`): (n_train, 28, 28) training images
        y_train (:obj:`np.array`): (n_train, 1) training labels
        x_test (:obj:`np.array`): (n_test, 28, 28) training images
        y_test (:obj:`np.array`y): (n_test, 1) training labels
        mapping (:obj:`np.array`): (62, 2) maps training labels (first column)
            to ascii codes (second column)
    """
    filepath = os.path.join(emnist_folder_path, EMNIST_FILENAME)

    if not (os.path.isdir(emnist_folder_path) and os.path.isfile(filepath)):
        if download:
            download_emnist(emnist_folder_path)
        else:
            error_msg = ("Folder {} or file {} does not exist "
                         "and download is deactivated").format(
                                emnist_folder_path, EMNIST_FILENAME)
            raise FileNotFoundError(error_msg)

    raw_emnist_data = loadmat(filepath)

    _logger.info("Loading train and test data from %s", filepath)

    # load train data
    x_train = raw_emnist_data["dataset"][0][0][0][0][0][0].astype(np.float32)
    x_train = x_train.reshape((x_train.shape[0], 28, 28), order='A')
    y_train = raw_emnist_data["dataset"][0][0][0][0][0][1]

    # load test data
    x_test = raw_emnist_data["dataset"][0][0][1][0][0][0].astype(np.float32)
    x_test = x_test.reshape((x_test.shape[0], 28, 28), order='A')
    y_test = raw_emnist_data["dataset"][0][0][1][0][0][1]

    # load mapping from label to chr
    mapping = raw_emnist_data["dataset"][0][0][2]

    return x_train, y_train, x_test, y_test, mapping


def download_emnist(emnist_folder_path):
    """
    Downloads, extracts and moves files to desired path

    Args:
        emnist_folder_path (str): folder for EMNIST data download and extraction
    """
    _logger.info("Data not found. Starting EMNIST Download from %s",
                 EMNIST_MATLAB_URL)
    if not os.path.isdir(emnist_folder_path):
        os.mkdir(emnist_folder_path)
        _logger.info("Target folder not found. Created %s", emnist_folder_path)

    emnist_matlab_zip_filepath = os.path.join(emnist_folder_path,
                                              'emnist_matlab.zip')
    req = requests.get(EMNIST_MATLAB_URL, stream=True)

    with open(emnist_matlab_zip_filepath, 'wb') as fp:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                fp.write(chunk)

    _logger.info("Download successful. Extracting and moving files...")
    with zipfile.ZipFile(emnist_matlab_zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(emnist_folder_path)

    # move
    matlab_folder_path = os.path.join(emnist_folder_path, 'matlab')
    for matlab_file in os.listdir(matlab_folder_path):
        shutil.move(os.path.join(matlab_folder_path, matlab_file),
                    os.path.join(emnist_folder_path, matlab_file))

    shutil.rmtree(matlab_folder_path)


def show_img(idx, x_train, y_train, x_test, y_test, mapping, mode='train'):
    """
    Shows the selected EMNIST image and its label

    Args:
        idx(int):
        x_train (:obj:`np.array`): (n_train, 28, 28) training images
        y_train (:obj:`np.array`): (n_train, 1) training labels
        x_test (:obj:`np.array`): (n_test, 28, 28) training images
        y_test (:obj:`np.array`y): (n_test, 1) training labels
        mapping (:obj:`np.array`): (62, 2) maps training labels (first column)
            to ascii codes (second column)
        mode (str): select image either from `train` or `test` dataset
    """
    if mode == 'train':
        img = x_train[idx]
        label = mapping[y_train[idx, 0]]
    else:
        img = x_test[idx]
        label = mapping[y_test[idx, 0]]
    plt.figure()
    plt.gray()
    plt.imshow(img)
    plt.show()
    print("Label: {}".format(label))


def show_train_progress(iteration, train_loss, test_loss, train_acc, test_acc):
    print(("Iteration: {:06d}, "
           "Loss (Train/Test): {:.3f} / {:.3f}, "
           "Accuracy: {:.3%} / {:.3%}").format(iteration,
                                               train_loss, test_loss,
                                               train_acc, test_acc))


def eval_serving_accuracy(n_examples, n_print_examples, request_url,
                          dataset='test'):
    """


    Args:
        n_examples (int): number of examples to pick and evaluate
        n_print_examples (int): number of examples to print img and classification
        request_url (str):
        dataset (str): `train` or `test` data to pick evaluation examples from
    """
    x_train, y_train, x_test, y_test, _ = load_emnist('emnist_data/')
    mapping = get_emnist_mapping()

    acc = 0

    if dataset == 'train':
        x_eval, y_eval = x_train, y_train
    else:
        x_eval, y_eval = x_test, y_test

    eval_img_indices = np.random.choice(np.arange(x_eval.shape[0]),
                                        n_examples,
                                        replace=False)

    for idx, test_img_idx in enumerate(eval_img_indices):
        test_img_flatten = x_eval[test_img_idx].reshape(1, 784) / 255
        test_img_payload = {"instances": test_img_flatten.tolist()}
        test_img_softmax_pred = requests.post(
                request_url, data=json.dumps(test_img_payload)
        )
        test_img_class_pred = np.argmax(
                test_img_softmax_pred.json()['predictions'])
        acc += (test_img_class_pred == y_eval[test_img_idx][0])

        # print the first 10 images with their true and predicted label
        if idx < n_print_examples:
            show_img(test_img_idx, x_train, y_train, x_test, y_test, mapping,
                     mode=dataset)
            print("Predicted Label: {}".format(mapping[test_img_class_pred]))

    print("Implementation Accuracy on {} test images: {:.2%}".format(
          n_examples, acc / n_examples))


def get_emnist_mapping():
    """
    Create Mapping for 62 classes from ASCII code to digit or character

    Returns:
        mapping (dict): maps ASCII integer to digit / character string
    """
    ascii_digits = list(range(48, 58))
    ascii_uppercase_letters = list(range(65, 65+26))
    ascii_lowercase_letters = list(range(97, 97+26))

    digits = [chr(value) for value in ascii_digits]
    uppercase_letters = [chr(value) for value in ascii_uppercase_letters]
    lowercase_letters = [chr(value) for value in ascii_lowercase_letters]

    mapping = dict(zip(range(10+2*26),
                       digits+uppercase_letters+lowercase_letters))

    return mapping
