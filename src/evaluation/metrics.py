"""
Evaluation metrics for OCR output quality.

Computes Character Error Rate (CER) and Word Error Rate (WER) using
the Hugging Face `evaluate` library and `jiwer`.
"""

import evaluate


class MetricsEvaluator:
    def __init__(self):
        """Initialize CER and WER metrics."""
        self.cer_metric = evaluate.load("cer")
        self.wer_metric = evaluate.load("wer")

    def compute(self, predictions, references):
        """
        Compute CER and WER for lists of predictions and references.

        :param predictions: List of predicted strings
        :param references: List of ground truth strings
        :return: dict with 'cer' and 'wer' scores (floats between 0 and 1+)
        """
        if not predictions or not references:
            return {"cer": 0.0, "wer": 0.0}

        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
            )

        # Filter out empty references (CER/WER is undefined for empty strings)
        valid_pairs = [
            (p, r) for p, r in zip(predictions, references) if r.strip()
        ]
        if not valid_pairs:
            return {"cer": 0.0, "wer": 0.0}

        valid_preds, valid_refs = zip(*valid_pairs)

        cer = self.cer_metric.compute(predictions=list(valid_preds), references=list(valid_refs))
        wer = self.wer_metric.compute(predictions=list(valid_preds), references=list(valid_refs))

        return {"cer": cer, "wer": wer}

    def compute_single(self, prediction, reference):
        """Compute CER and WER for a single prediction/reference pair."""
        return self.compute([prediction], [reference])


if __name__ == "__main__":
    evaluator = MetricsEvaluator()

    # Quick test
    preds = ["hello world", "this is a test", "the quick brown fox"]
    refs = ["hello world", "this is test", "the quick brown fox"]
    results = evaluator.compute(preds, refs)
    print(f"CER: {results['cer']:.4f}")
    print(f"WER: {results['wer']:.4f}")
