import torch

from pl_modules.model import ImageClassifier


INPUT_CKPT = "/Users/vl.naumov/Desktop/courses/mlops_course/cats-and-dogs/models/epoch=04-val_loss=0.5773.ckpt"
OUTPUT_PTH = "model_weights.pth"

print(f"Loading Lightning checkpoint: {INPUT_CKPT}")

model = ImageClassifier.load_from_checkpoint(INPUT_CKPT, map_location="cpu")

state_dict = model.state_dict()

print("Saving weights-only file...")
torch.save(state_dict, OUTPUT_PTH)

print("Done!")
