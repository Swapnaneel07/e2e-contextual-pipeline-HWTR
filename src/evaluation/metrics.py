import evaluate

class MetricsEvaluator:
    def __init__(self):
        """
        Initialize the Hugging Face evaluate module for Character Error Rate (CER)
        and Word Error Rate (WER).
        """
        self.cer_metric = evaluate.load("cer")
        self.wer_metric = evaluate.load("wer")

    def compute_metrics(self, predictions, references):
        """
        Compute CER and WER for a list of predictions and references.
        :param predictions: List of predicted strings
        :param references: List of ground truth strings
        :return: Dictionary containing average CER and WER
        """
        if not predictions or not references:
            return {"cer": 0.0, "wer": 0.0}
            
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        cer_score = self.cer_metric.compute(predictions=predictions, references=references)
        wer_score = self.wer_metric.compute(predictions=predictions, references=references)

        return {
            "cer": cer_score,
            "wer": wer_score
        }

if __name__ == "__main__":
    evaluator = MetricsEvaluator()
    preds = ["hello world", "this is a test"]
    refs = ["hello world", "this is test"]
    results = evaluator.compute_metrics(preds, refs)
    print("Metrics:", results)
