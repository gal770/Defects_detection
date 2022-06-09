from torch import nn

from xception import build_xception_backbone


def get_xception_based_model() -> nn.Module:
    """Return an Xception-Based network.

    (1) Build an Xception  backbone and hold it as `custom_network`.
    (2) Override `custom_network`'s fc attribute with the binary
    classification head."""

    xception_backbone = build_xception_backbone()
    xception_backbone.fc = nn.Sequential(
        nn.Linear(2048, 1000),
        nn.ReLU(),
        nn.Linear(1000, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 2))

    return xception_backbone

if __name__ == '__main__':
    get_xception_based_model()