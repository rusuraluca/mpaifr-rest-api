class MetricsRecorder:
    """
    A utility class for recording and computing loss and accuracy.
    """
    def __init__(self):
        self.total_samples = 0
        self.total_loss = 0.0
        self.total_correct = 0

    def reset(self):
        self.total_samples = 0
        self.total_loss = 0.0
        self.total_correct = 0

    def gulp(self, total_samples, total_loss, total_correct):
        self.total_samples += total_samples
        self.total_loss += total_samples * total_loss
        self.total_correct += int(total_samples * total_correct)

    def excrete(self):
        self.total_loss = self.total_loss / self.total_samples
        self.total_correct = self.total_loss / self.total_samples
        return self

    def result(self):
        return f"{self.total_samples}, {self.total_loss:.4f}, {self.total_correct:.4f}"
