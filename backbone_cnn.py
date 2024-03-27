import torch.nn as nn


class LayersConfig(object):
    """
    Layers is a Singleton class that defines one instance of a list with the type of layers that the Backbone CNN uses.
    """
    _instance = None

    def __init__(self):
        """
        Raise an exception in __init__() to make normal object instantiation impossible.
        """
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        """
        Callers are instructed to use the instance() class method,
        which creates the object once and returns the object.
        :return: list of layers
        """
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance = [
                (3, 64),            # BatchNorm2d, Conv2d, ReLU
                (64, 64),           # BatchNorm2d, Conv2d, ReLU
                "M",                # MaxPool2d

                (64, 64, 3),        # ResidualLayer

                (64, 128),          # BatchNorm2d, Conv2d, ReLU
                "M",                # MaxPool2d

                (128, 128, 4),      # ResidualLayer

                (128, 256),         # BatchNorm2d, Conv2d, ReLU
                "M",                # MaxPool2d

                (256, 256, 10),     # ResidualLayer

                (256, 512),         # BatchNorm2d, Conv2d, ReLU
                "M",                # MaxPool2d

                (512, 512, 3)       # ResidualLayer
            ]
        return cls._instance


class MaxPool:
    """
    A wrapper class for PyTorch's MaxPool2d layer.
    """
    def __init__(self):
        """
        Initializes the MaxPool class by creating a MaxPool2d layer with a kernel size of 2.

        This specific kernel size is chosen for its common use in downsampling operations within CNNs,
        effectively reducing the spatial dimensions of the input feature map by half.
        """
        self._object = nn.MaxPool2d(kernel_size=2)

    def get_object(self):
        """
        Retrieves the encapsulated MaxPool2d layer.

        :return: MaxPool2d layer with a kernel size of 2
        """
        return self._object


class BatchNormConvReLU:
    """
    A composite layer class that encapsulates a sequence of operations common in convolutional neural networks (CNNs):
    batch normalization, convolution, and ReLU activation.

    This class creates a sequence of layers that is frequently used to preprocess inputs before passing them
    through subsequent layers of a neural network. By combining these layers, it aims to improve training stability,
    accelerate convergence, and enable the use of higher learning rates.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initializes the composite layer with the specified input and output channels.

        :param in_channels: number of channels in the input tensor
        :param out_channels: number of filters in the convolution layer,
                             and hence the number of channels in the output tensor
        """
        self._object = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def get_object(self):
        """
        Retrieves the sequential container holding the batch normalization, convolution, and ReLU layers.

        :return: encapsulated sequence of layers
        """
        return self._object


class ResidualBlock(nn.Module):
    """
    Residual Block with 3 stacked units of 3x3 [BatchNorm2d + Conv2d + ReLU].
    Originally introduced in
    Wang, Yitong & Gong, Dihong & Zhou, Zheng & Ji, Xing & Wang, Hao & Li, Zhifeng & Liu, Wei & Zhang, Tong.
    (2018). Orthogonal Deep Features Decomposition for Age-Invariant Face Recognition.
    """
    def __init__(self, input_channels, output_channels, block_size=3):
        """
        Initializes the ResidualBlock with specific input and output channel sizes.
        :param input_channels: depth of the input feature map
        :param output_channels: depth of  the output feature map after processing
        """
        super(ResidualBlock, self).__init__()
        mods = []
        for _ in range(block_size):
            mods.extend([
                nn.BatchNorm2d(input_channels),
                nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True)
            ])
            input_channels = output_channels
        self.sequential = nn.Sequential(*mods)

    def forward(self, in_tensor):
        """
        Passes the input through the convolutional sequence and then added back to the original
        input tensor, implementing the residual connection.
        :param in_tensor: input tensor to be passed through residual block
        :return: output tensor
        """
        return self.sequential(in_tensor) + in_tensor


class LayersBuilder(nn.Module):
    """
    LayersBuilder is a Builder class that build the list of layers for the Backbone CNN.
    """
    def __init__(self):
        super(LayersBuilder, self).__init__()
        self.layers = nn.ModuleList()

    def construct(self, layers_config):
        """
        Build the list of layers from the provided configuration
        :param layers_config: layers configuration
        :return:
        """
        for config in layers_config:
            if config == "M":
                self.layers.append(
                    MaxPool().get_object()
                )
            elif len(config) == 2:
                in_channels, out_channels = config
                self.layers.append(
                    BatchNormConvReLU(in_channels, out_channels).get_object()
                )
            elif len(config) == 3:
                in_channels, out_channels, repetitions = config
                for _ in range(repetitions):
                    self.layers.append(
                        ResidualBlock(in_channels, out_channels)
                    )
            else:
                raise ValueError(config)

        return self.layers


class BackboneCNN(nn.Module):
    """
    Orthogonal Embedding CNN (OECNN), 64-layer CNN which consists of 4 stages
    with respectively 3, 4, 10, 3 stacked residual blocks
    and a final Fully-Connected (FC) layer that outputs the initial face features of 512 dimension.
    Originally introduced in
    Wang, Yitong & Gong, Dihong & Zhou, Zheng & Ji, Xing & Wang, Hao & Li, Zhifeng & Liu, Wei & Zhang, Tong.
    (2018). Orthogonal Deep Features Decomposition for Age-Invariant Face Recognition.
    """
    def __init__(self, initializer, dropout_probability=0):
        """
        Initializes the OECNN model with a specific weight initializer and dropout probability.
        :param initializer: dictionary containing
                            an initialization method ('method') and parameters ('params') for that method
        :param dropout_probability: probability of an element to be zeroed in the dropout layer
                                    prevents overfitting
        """
        super(BackboneCNN, self).__init__()

        layers_config = LayersConfig.instance()
        layers = LayersBuilder().construct(layers_config)

        self.model = nn.Sequential(
            *layers,
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Dropout(dropout_probability),
            nn.Linear(512*7*6, 512),
            nn.BatchNorm1d(512)
        )

        self._initialize_weights(initializer)

    def _initialize_weights(self, initializer):
        """
        Initializes the weights and biases of the model's layers.
        :param initializer: dictionary containing
                            an initialization method ('method') and parameters ('params') for that method
        """
        if initializer is not None:
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    initializer['method'](module.weight, **initializer['params'])
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

                elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)

                elif isinstance(module, nn.Linear):
                    initializer['method'](module.weight, **initializer['params'])
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    def forward(self, in_tensor):
        """
        Passes the input through the sequential container model,
        then it is renormalized to ensure that the output tensor's 2-norm does not exceed the 1e-5 threshold
        then it is scaled by 1e5
        all to ensure that the output is within a certain range to improve the training stability.
        :param in_tensor: input tensor to be passed through the model
        :return: output tensor
        """
        return self.model(in_tensor).renorm(2, 0, 1e-5).mul(1e5)

    def __str__(self):
        return "Backbone CNN"
