import torch
import torch.nn as nn
import torch.nn.functional as functional
from backbone_cnn import BackboneCNN
from loss_functions import MarginLossFactory
from rfm import RFM
from utils.model_utils import accuracy


class DALR(nn.Module):
    """
    DAL Regularizer
    Originally introduced in
    Hao Wang, Dihong Gong, Zhifeng Li, and Wei Liu.
    Decorrelated adversarial learning for age-invariant face recognition, 2019.
    """
    def __init__(self, input_dimension):
        """
        Initializes the DAL regularizer module with linear layers for projecting age-dependent and identity-dependent
        features.
        :param input_dimension: size of the input feature vector for both age-dependent and identity-dependent features
        """
        super(DALR, self).__init__()
        self.id_predictor = nn.Linear(input_dimension, 1, bias=False)
        self.age_predictor = nn.Linear(input_dimension, 1, bias=False)
        self.gender_predictor = nn.Linear(input_dimension, 1, bias=False)

    def forward(self, id_features, age_features, gender_features):
        """
        Computes the decorrelation loss for a given batch of age-dependent and identity-dependent features.
        :param gender_features:
        :param id_features: tensor containing the identity-dependent features of the input data
        :param age_features: tensor containing the age-dependent features of the input data
        :return: scalar tensor representing the decorrelation loss,
                 which reduces the statistical dependency between age-dependent and identity-dependent features
        """
        id_predictions = self.id_predictor(id_features)
        age_predictions = self.age_predictor(age_features)
        gender_predictions = self.gender_predictor(gender_features)

        # Calculate mean and variance for predictions
        id_mean = id_predictions.mean(dim=0)
        age_mean = age_predictions.mean(dim=0)
        gender_mean = gender_predictions.mean(dim=0)

        id_var = id_predictions.var(dim=0) + 1e-6
        age_var = age_predictions.var(dim=0) + 1e-6
        gender_var = gender_predictions.var(dim=0) + 1e-6

        # Calculate the squared correlation coefficient between each pair of features
        id_age_corr = ((age_predictions - age_mean) * (id_predictions - id_mean)).mean(dim=0).pow(2) / (
                    age_var * id_var)
        id_gender_corr = ((gender_predictions - gender_mean) * (id_predictions - id_mean)).mean(dim=0).pow(2) / (
                    gender_var * id_var)
        age_gender_corr = ((age_predictions - age_mean) * (gender_predictions - gender_mean)).mean(dim=0).pow(2) / (
                    age_var * gender_var)

        # You could average the correlation coefficients for a combined measure
        correlation_coefficient = (id_age_corr + id_gender_corr + age_gender_corr) / 3
        return correlation_coefficient


class DAL(nn.Module):
    """
    Decorrelated Adversarial Learning (DAL) approach integrates
    a backbone network (OECNN) for feature extraction,
    a Residual Factorization Module (RFM) to separate age and identity features,
    utilizes specific margin-based loss functions (e.g., CosFace, ArcFace) for enhancing discriminative power,
    and it employs a DAL Regularizer (DALR) to minimize correlation between age and identity features.
    Originally introduced in
    Hao Wang, Dihong Gong, Zhifeng Li, and Wei Liu.
    Decorrelated adversarial learning for age-invariant face recognition, 2019.
    """
    def __init__(self, loss_head, num_classes, embedding_dimension=512, initializer=None):
        """
        Initializes the DAL model with specified configurations.
        :param loss_head: type of margin-based loss function to use for training
        :param num_classes: number of identity classes in the dataset
        :param embedding_dimension: dimensionality of the feature embeddings produced by the OECNN
        :param initializer: initialization method for the OECNN
        """
        super(DAL, self).__init__()

        self.backboneCNN = BackboneCNN(initializer)
        self.margin_loss = MarginLossFactory().get_margin_loss(loss_head, num_classes, embedding_dimension)
        self.DALR = DALR(embedding_dimension)
        self.RFM = RFM(embedding_dimension)

        self.age_classifier = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dimension, 8)
        )

        self.gender_classifier = nn.Sequential(
            nn.Linear(embedding_dimension, 2),
            nn.Softmax(dim=1)
        )

        self.id_criterion = nn.CrossEntropyLoss()
        self.age_criterion = nn.CrossEntropyLoss()
        self.gender_criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, labels=None, age_groups=None, genders=None, return_embeddings=False):
        """
        Forward pass through the DAL model.
        :param genders:
        :param inputs: tensor input images
        :param labels: ground truth labels for identity classification
        :param age_groups: ground truth labels for age group classification
        :param return_embeddings: if True, returns normalized identity embeddings instead of losses and accuracies
        :return: depending on return_embeddings,
                 either normalized identity embeddings
                 or a tuple containing identity loss, identity accuracy, age loss, age accuracy, and decorrelation loss
        """
        embeddings = self.backboneCNN(inputs)
        id_embeddings, age_embeddings, gender_embeddings = self.RFM(embeddings)

        if return_embeddings:
            return functional.normalize(id_embeddings)

        id_logits = self.margin_loss(id_embeddings, labels)
        id_loss = self.id_criterion(id_logits, labels)
        id_accuracy = accuracy(torch.max(id_logits, dim=1)[1], labels)

        age_logits = self.age_classifier(age_embeddings)
        age_loss = self.age_criterion(age_logits, age_groups)
        age_accuracy = accuracy(torch.max(age_logits, dim=1)[1], age_groups)

        gender_logits = self.gender_classifier(gender_embeddings)
        gender_loss = self.gender_criterion(gender_logits, genders)
        gender_accuracy = accuracy(torch.max(gender_logits, dim=1)[1], genders)

        cano_cor = self.DALR(id_embeddings, age_embeddings, gender_embeddings)

        return id_loss, id_accuracy, age_loss, age_accuracy, gender_loss, gender_accuracy, cano_cor
