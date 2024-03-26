import torch.nn as nn


class RFM(nn.Module):
    """
    Residual factorization Module (RFM) to map the initial face features
    to form the age-dependent feature through 2 [FC +ReLU]
    and the gender-dependent feature through 2 [FC +ReLU],
    and the residual part is regarded as the identity-dependent feature.
    """
    def __init__(self, in_dimension):
        """
        Initializes the Residual Factorization Module with the given input dimensionality.
        :param in_dimension: size of the input feature vector
        """
        super(RFM, self).__init__()
        self.age_transform = nn.Sequential(
            nn.Linear(in_dimension, in_dimension),
            nn.ReLU(inplace=True),
            nn.Linear(in_dimension, in_dimension),
            nn.ReLU(inplace=True)
        )

        self.gender_transform = nn.Sequential(
            nn.Linear(in_dimension, in_dimension),
            nn.ReLU(inplace=True),
            nn.Linear(in_dimension, in_dimension),
            nn.ReLU(inplace=True)
        )

    def forward(self, in_tensor):
        """
        Passes the input through the sequence of transformations defined.
        :param in_tensor:
        :return:
        """
        age_features = self.age_transform(in_tensor)
        gender_features = self.gender_transform(in_tensor)
        identity_features = in_tensor - age_features - gender_features

        return identity_features, age_features, gender_features
