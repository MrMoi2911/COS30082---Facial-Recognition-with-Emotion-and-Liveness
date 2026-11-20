# Trained models

No trained weights are included in this source package.

After Colab training, extract each exported result ZIP into the repo root. This creates:

- `models/face_classification/best.pt`
- `models/emotion/best.pt`
- `reports/face_classification/verification/thresholds.json`

The actual files are ignored by Git. Share the resulting weights through the group's Drive
folder or model releases. Add the real model link and SHA-256 to the main README after training.
Keep `last.pt` in Drive to resume training. Do not use `last.pt` instead of the validation-selected
`best.pt` for final evaluation unless you explicitly justify that change.
