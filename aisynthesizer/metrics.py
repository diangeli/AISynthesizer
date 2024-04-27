from torchmetrics.functional.classification import binary_accuracy, binary_auroc, binary_precision


class Metrics:
    @staticmethod
    def get_accuracy(y_pred, midi_labels, threshold):
        return binary_accuracy(y_pred, midi_labels, threshold = threshold)

    @staticmethod
    def get_precision(y_pred, midi_labels, threshold):
        return binary_precision(y_pred, midi_labels, threshold)

    @staticmethod
    def get_auroc(y_pred, midi_labels):
        return binary_auroc(y_pred, midi_labels)

    @staticmethod
    def get_all_metrics(y_pred, midi_labels):
        return {
            "accuracy": binary_accuracy(y_pred, midi_labels),
            "precision": binary_precision(y_pred, midi_labels),
            "auroc": binary_auroc(y_pred, midi_labels),
        }