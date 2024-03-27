import torch
import torch.nn as nn
import torch.nn.functional as functional


class MarginLossFactory:
    """
    Factory design pattern class for creating margin loss objects based on a given loss head type.

    This class simplifies the process of initializing different types of margin losses
    by encapsulating the decision-making process within a single method.
    """
    @staticmethod
    def get_margin_loss(loss_head, num_classes, embedding_dimension):
        if loss_head.lower() == 'cosface':
            return CosFaceMarginLoss(num_classes, embedding_dimension)

        elif loss_head.lower() == 'v2cosface':
            return V2CosFaceMarginLoss(num_classes, embedding_dimension)

        elif loss_head.lower() == 'arcface':
            return ArcFaceMarginLoss(num_classes, embedding_dimension)

        else:
            raise ValueError("Unsupported loss head.")


class ArcFaceMarginLoss(nn.Module):
    def __init__(self, num_classes, size_embedding, scale=64.0, margin=0.3):
        super(ArcFaceMarginLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.embedding_weights = nn.Parameter(
            torch.Tensor(num_classes, size_embedding)
        )
        nn.init.xavier_uniform_(self.embedding_weights)

    def forward(self, features, target):
        logits = functional.linear(functional.normalize(features), functional.normalize(self.weight))
        if not self.training:
            return logits
        return logits.scatter(
            1,
            target.view(-1, 1),
            (logits.gather(1, target.view(-1, 1)).acos() + self.margin).cos()
        ).mul(self.scale)


class CosFaceMarginLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, scale=64.0, margin=0.35):
        super(CosFaceMarginLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(
            torch.Tensor(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        coses = functional.linear(functional.normalize(embeddings), functional.normalize(self.weight))
        if not self.training:
            return coses
        return coses.scatter_add(
            1,
            labels.view(-1, 1),
            coses.new_full(labels.view(-1, 1).size(), -self.margin)
        ).mul(self.scale)


class V2CosFaceMarginLoss(nn.Module):
    def __init__(self, number_classes, size_embedding=512, scale=64.0, margin=0.35):
        super(V2CosFaceMarginLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.classifier = nn.Linear(size_embedding, number_classes, bias=False)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, features, labels):
        coses = self.classifier(features.renorm(2, 0, 1e-5).mul(1e5))
        if not self.training:
            return coses
        return coses.scatter_add(
            1,
            labels.view(-1, 1),
            coses.new_full(labels.view(-1, 1).size(), -self.margin)
        ).mul(self.scale)
