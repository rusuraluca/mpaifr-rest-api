import torch.nn as nn


class RFM(nn.Module):
    """
    Residual factorization Module (RFM) to map the initial face features
    to form the age-dependent feature through sequential 2 [FC +ReLU]
    and the gender-dependent feature through sequential 2 [FC +ReLU],
    and the residual part is regarded as the identity-dependent feature.
    """
    def __init__(self, input_dimension):
        """
        Initializes the Residual Factorization Module with the given input dimensionality.
        :param input_dimension: size of the input feature vector
        """
        super(RFM, self).__init__()
        self.age_transform = nn.Sequential(
            nn.Linear(input_dimension, input_dimension),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Linear(input_dimension, input_dimension),
            nn.ELU(alpha=1.0, inplace=True)
        )

        self.gender_transform = nn.Sequential(
            nn.Linear(input_dimension, input_dimension),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Linear(input_dimension, input_dimension),
            nn.ELU(alpha=1.0, inplace=True)
        )

    def forward(self, input_tensor):
        """
        Passes the input through the sequence of transformations defined.
        :param input_tensor: input feature vector
        :return:
        """
        age_features = self.age_transform(input_tensor)
        gender_features = self.gender_transform(input_tensor)
        id_features = input_tensor - age_features - gender_features

        return id_features, age_features, gender_features
