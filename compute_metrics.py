import csv
import os
from jiwer import wer, cer

def load_csv(csv_path):
    """
    Load CSV file into a dictionary {filename: text}

    Args:
        csv_path: Path to CSV file

    Returns:
        dict: {filename: text}
    """
    data = {}
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            data[row['image_filename']] = row['text'] if 'text' in row else row['predicted_text']
    return data

def compute_metrics(ground_truth, predictions):
    """
    Compute WER and CER for predictions vs ground truth.

    Args:
        ground_truth: Dict {filename: true_text}
        predictions: Dict {filename: predicted_text}

    Returns:
        dict: Metrics
    """
    true_texts = []
    pred_texts = []

    for filename in ground_truth:
        if filename in predictions:
            true_texts.append(ground_truth[filename])
            pred_texts.append(predictions[filename])

    if not true_texts:
        return {"error": "No matching filenames found"}

    wer_score = wer(true_texts, pred_texts)
    cer_score = cer(true_texts, pred_texts)

    return {
        "WER": wer_score,
        "CER": cer_score,
        "num_samples": len(true_texts)
    }

def main():
    # Paths
    ground_truth_csv = 'input/gnhk_dataset/test_processed.csv'
    pretrained_pred_csv = 'pretrained_model_predictions.csv'
    fine_tuned_pred_csv = 'fine_tuned_model_predictions.csv'

    # Load data
    print("Loading ground truth...")
    ground_truth = load_csv(ground_truth_csv)

    print("Loading pretrained predictions...")
    pretrained_pred = load_csv(pretrained_pred_csv)

    print("Loading fine-tuned predictions...")
    fine_tuned_pred = load_csv(fine_tuned_pred_csv)

    # Compute metrics
    print("\nComputing metrics for pretrained model...")
    pretrained_metrics = compute_metrics(ground_truth, pretrained_pred)

    print("Computing metrics for fine-tuned model...")
    fine_tuned_metrics = compute_metrics(ground_truth, fine_tuned_pred)

    # Print results
    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)

    print(f"\nPretrained Model (microsoft/trocr-small-handwritten):")
    if "error" in pretrained_metrics:
        print(f"  Error: {pretrained_metrics['error']}")
    else:
        print(f"  Word Error Rate (WER): {pretrained_metrics['WER']:.4f}")
        print(f"  Character Error Rate (CER): {pretrained_metrics['CER']:.4f}")
        print(f"  Number of samples: {pretrained_metrics['num_samples']}")

    print(f"\nFine-tuned Model:")
    if "error" in fine_tuned_metrics:
        print(f"  Error: {fine_tuned_metrics['error']}")
    else:
        print(f"  Word Error Rate (WER): {fine_tuned_metrics['WER']:.4f}")
        print(f"  Character Error Rate (CER): {fine_tuned_metrics['CER']:.4f}")
        print(f"  Number of samples: {fine_tuned_metrics['num_samples']}")

    # Comparison
    if "error" not in pretrained_metrics and "error" not in fine_tuned_metrics:
        print(f"\nImprovement:")
        wer_improvement = pretrained_metrics['WER'] - fine_tuned_metrics['WER']
        cer_improvement = pretrained_metrics['CER'] - fine_tuned_metrics['CER']
        print(f"  WER improvement: {wer_improvement:.4f} ({wer_improvement*100:.2f}%)")
        print(f"  CER improvement: {cer_improvement:.4f} ({cer_improvement*100:.2f}%)")

if __name__ == '__main__':
    main()