import os
import torch
from lid_model import LIDModel

TRANSCRIPT = "transcript_final.txt"
TRANSLATION = "bhojpuri_translation.txt"
OUTPUT_AUDIO = "results/output_LRL_cloned.wav"
LID_MODEL_PATH = "lid_model.pth"

def run_translation():
    print("\n[STEP 1] Translation to Bhojpuri")
    if not os.path.exists(TRANSLATION):
        print("Running translation...")
        os.system("python translate.py")
    else:
        print("Translation file already exists")

def run_tts():
    print("\n[STEP 2] Speech Synthesis")
    if not os.path.exists(OUTPUT_AUDIO):
        print("Generating speech...")
        os.system("python synthetize_gtts.py")
    else:
        print("Audio already exists")

def load_lid_model():
    print("\n[STEP 3] Loading LID Model")

    model = LIDModel()
    if os.path.exists(LID_MODEL_PATH):
        model.load_state_dict(torch.load(LID_MODEL_PATH))
        model.eval()
        print("LID model loaded successfully")
    else:
        print("WARNING: LID model not found")

    return model

def run_evaluation():
    print("\n[STEP 4] Running Evaluation Metrics")
    os.system("python evaluate_all.py")

def main():
    print("=" * 50)
    print("RUNNING FULL SPEECH PIPELINE")
    print("=" * 50)

    run_translation()
    run_tts()
    load_lid_model()
    run_evaluation()

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()