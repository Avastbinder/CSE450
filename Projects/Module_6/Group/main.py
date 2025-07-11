import torch
from author_rnn import CharLSTM

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the saved model
    model_path = "checkpoints/best_model"
    model, char_to_idx, idx_to_char = CharLSTM.load_model(model_path, device=device)

    print("Model loaded successfully!\n")

    while True:
        # User prompt
        start_seq = input("Enter a starting sequence (or type 'quit' to exit): ")
        if start_seq.lower() == 'quit':
            break

        # Generate text
        length = int(input("How many characters to generate? (e.g., 500): "))
        temperature = float(input("Enter temperature (suggested: 0.7 to 1.0): "))

        print("\nGenerating text...")
        output = model.generate_text(
            start_seq=start_seq,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            length=length,
            temperature=temperature,
            device=device
        )

        print("\n--- Generated Text ---")
        print(output)
        print("\n")

if __name__ == "__main__":
    main()
