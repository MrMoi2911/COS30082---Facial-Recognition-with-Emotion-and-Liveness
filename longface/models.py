import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision import transforms as T


class FaceClassifier(nn.Module):
    def __init__(self, num_classes, embedding_dim=128, pretrained=False,
                 train_backbone="layer4"):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.embedding = nn.Linear(512, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        if train_backbone not in {"all", "layer4"}:
            raise ValueError("train_backbone must be all or layer4")
        self.train_backbone = train_backbone
        if train_backbone == "layer4":
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad = name.startswith("layer4.")

    def train(self, mode=True):
        super().train(mode)
        if mode and self.train_backbone == "layer4":
            self.backbone.eval()
            self.backbone.layer4.train()
        return self

    def encode(self, images):
        return F.normalize(self.embedding(self.backbone(images)), p=2, dim=1)

    def forward(self, images):
        features = self.embedding(self.backbone(images))
        return self.classifier(features)


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        layers = []
        channels = [1, 32, 64, 128]
        for source, target in zip(channels, channels[1:]):
            layers.extend([nn.Conv2d(source, target, 3, padding=1),
                           nn.BatchNorm2d(target), nn.ReLU(inplace=True), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((3, 3)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 3 * 3, 128),
                                        nn.ReLU(inplace=True), nn.Dropout(0.4),
                                        nn.Linear(128, num_classes))

    def forward(self, images):
        return self.classifier(self.features(images))


def face_transform(image_size=224, training=False):
    operations = [T.Resize((image_size, image_size))]
    if training:
        operations.extend([T.RandomHorizontalFlip(),
                           T.ColorJitter(brightness=0.15, contrast=0.15)])
    operations.extend([T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return T.Compose(operations)


def emotion_transform(training=False):
    operations = [T.Grayscale(num_output_channels=1), T.Resize((48, 48))]
    if training:
        operations.extend([T.RandomHorizontalFlip(), T.RandomRotation(8)])
    operations.extend([T.ToTensor(), T.Normalize([0.5], [0.5])])
    return T.Compose(operations)


def build_model(config, pretrained=False):
    if config["task"] == "face":
        return FaceClassifier(len(config["classes"]), config["embedding_dim"],
                              pretrained, config["train_backbone"])
    if config["task"] == "emotion":
        return EmotionCNN(len(config["classes"]))
    raise ValueError("Unknown task in model config")


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format. See docs/TEAM_CONTRACT.md")
    model = build_model(checkpoint["config"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint["config"]
