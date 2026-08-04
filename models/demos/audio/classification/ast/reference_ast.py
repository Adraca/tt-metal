import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import urllib.request

def main():
    model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
    print(f"Loading feature extractor and model for {model_id}...")
    
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_id)
    model = ASTForAudioClassification.from_pretrained(model_id)
    
    model.eval()

    # Create dummy audio data (1 second of random noise at 16kHz)
    # The AST model expects audio at 16kHz
    print("Generating dummy audio data...")
    dummy_audio = torch.randn(16000)

    # Preprocess the audio
    inputs = feature_extractor(dummy_audio, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values
    print(f"Input spectrogram shape: {input_values.shape}") # Should be [1, 1024, 128]

    print("Running PyTorch CPU forward pass...")
    with torch.no_grad():
        outputs = model(input_values)
        logits = outputs.logits

    print(f"Logits shape: {logits.shape}") # Should be [1, 527]
    print(f"First 10 logits: {logits[0, :10].numpy()}")
    
    predicted_class_idx = logits.argmax(-1).item()
    print(f"Predicted class: {model.config.id2label[predicted_class_idx]}")

if __name__ == "__main__":
    main()
