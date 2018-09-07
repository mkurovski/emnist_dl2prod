"""
Model classes for PyTorch and TensorFlow
"""
__author__ = "Marcel Kurovski"
__copyright__ = "Marcel Kurovski"
__license__ = "mit"


import tensorflow as tf
import torch
import torch.nn.functional as F


class LinearImgClassifier(torch.nn.Module):
    """Shallow neural network model for PyTorch
    that returns softmax class probabilities

    Attributes:
        n_input_feat (int): number of input features
        n_classes (int): number of output classes
    """
    def __init__(self, n_input_feat, n_classes):
        super(LinearImgClassifier, self).__init__()
        self.linear = torch.nn.Linear(n_input_feat, n_classes)

    def forward(self, x):
        y_pred = F.softmax(self.linear(x), dim=1)
        return y_pred


class DNNImgClassifier(torch.nn.Module):
    """Deep neural network model with 2 hidden layers for PyTorch
    that returns softmax class probabilities

    Attributes:
        n_input_feat (int): number of input features
        n_hidden_1 (int): number of units for the 1st hidden layer
        n_hidden_2 (int): number of units for the 2nd hidden layer
        n_classes (int): number of output classes

    """
    def __init__(self, n_input_feat, n_hidden_1, n_hidden_2, n_classes):
        super(DNNImgClassifier, self).__init__()
        self.linear_1 = torch.nn.Linear(n_input_feat, n_hidden_1)
        self.linear_2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.linear_3 = torch.nn.Linear(n_hidden_2, n_classes)

    def forward(self, x):
        logit_1 = self.linear_1(x)
        activ_1 = F.elu(logit_1)
        logit_2 = self.linear_2(activ_1)
        activ_2 = F.elu(logit_2)
        logit_3 = self.linear_3(activ_2)
        y_pred = F.softmax(logit_3, dim=1)
        return y_pred


class Model:
    """Model encapsulates the trained EMNIST neural network classifier

    Attributes:
        graph (:obj:`tf.Graph`): loaded TensorFlow graph
        sess (:obj:`tf.Session`): TensorFlow session object
    """
    def __init__(self, path):
        """Create model instance

        Args:
            path (str): path to the folder from which to load
                        model protobuf and variables
        """
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            tf.saved_model.loader.load(self.sess,
                                       [tf.saved_model.tag_constants.SERVING],
                                       path)

            self.input = self.graph.get_tensor_by_name(
                    "flattened_rescaled_img_28x28:0")
            self.output = self.graph.get_tensor_by_name("Softmax:0")

    def run(self, img):
        """Run the model on normalized and flattened input data
        and return softmax scores

        Args:
            img (:obj:`np.array`): (M, 784) array with M examples (rows)
                each containing a flattened (28, 28) grayscale image

        Returns:
            scores (:obj:`np.array`): (M, 62) array with softmax activation
                values for each instance and class
        """
        scores = self.sess.run(self.output, {self.input: img})

        return scores
