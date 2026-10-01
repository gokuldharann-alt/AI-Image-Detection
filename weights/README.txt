Trained weights live here (git-ignored in real projects).

Expected files after training:
  cifake_efficientnet_b0.pt    ← fine-tuned PyTorch checkpoint (used by app.py)
  cifake_efficientnet_b0.onnx  ← exported graph (auto-used when USE_ONNX=1)

How to produce them:
  1. Download CIFAKE from Kaggle (birdy654/cifake-real-and-ai-generated-synthetic-images)
     and extract to ./data/CIFAKE/{train,test}/{REAL,FAKE}/
  2. python train.py --data ./data/CIFAKE --epochs 12
  3. python export_onnx.py   (optional but recommended for CPU speed)

Without these files the app still runs in DEMO mode on ImageNet base weights
(predictions will NOT be accurate — train first!).
