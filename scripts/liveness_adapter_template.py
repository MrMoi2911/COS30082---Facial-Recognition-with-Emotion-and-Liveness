"""Copy this file into a teammate module and connect a genuinely trained model."""

from longface.liveness import LivenessResult


class TrainedLivenessAdapter:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        # Load your trained model, preprocessing, label mapping, and threshold here.

    def predict(self, face_rgb):
        # face_rgb is an RGB uint8 face crop. Apply your model's saved preprocessing.
        return LivenessResult("unavailable", None, "Adapter template has no trained model")


def create_liveness(checkpoint_path):
    return TrainedLivenessAdapter(checkpoint_path)
