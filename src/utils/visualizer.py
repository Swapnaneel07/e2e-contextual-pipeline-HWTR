import cv2
import matplotlib.pyplot as plt

def visualize_results(image_path, text_regions, raw_text, corrected_text=None, save_path=None):
    """
    Visualize the detection bounding boxes and the transcription results.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes
    for idx, region in enumerate(text_regions):
        x, y, w, h = region["bbox"]
        cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image_rgb, str(idx+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.figure(figsize=(15, 10))
    
    # Show Image
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Detected Text Regions")
    plt.axis("off")
    
    # Show Text
    plt.subplot(1, 2, 2)
    plt.axis("off")
    
    y_pos = 0.9
    plt.text(0.05, y_pos, "Raw OCR Text:", fontsize=14, fontweight="bold", color="blue")
    
    # Wrap text for visualization
    import textwrap
    raw_wrapped = textwrap.fill(raw_text, width=60)
    plt.text(0.05, y_pos - 0.1, raw_wrapped, fontsize=12, verticalalignment='top')
    
    if corrected_text:
        y_pos -= 0.5
        plt.text(0.05, y_pos, "LLM Corrected Text:", fontsize=14, fontweight="bold", color="green")
        cor_wrapped = textwrap.fill(corrected_text, width=60)
        plt.text(0.05, y_pos - 0.1, cor_wrapped, fontsize=12, verticalalignment='top')

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    pass
