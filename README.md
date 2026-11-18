

\# CAFA-TTA



Confidence-Aware Feature Alignment for Test-Time Adaptation

in Cross-Domain Medical Image Classification.



\## Research Direction



\- Medical Computer Vision

\- CUDA-Accelerated Deep Learning

\- Test-Time Adaptation

\- Confidence-Aware Feature Alignment

\- Continual Learning



\## Hardware



NVIDIA RTX 3050 Laptop GPU - 4GB VRAM



\## Environment



\- Python 3.12

\- PyTorch 2.10.0

\- CUDA 12.8



\## Dataset



PathMNIST:

\- Training: 89,996 images

\- Validation: 10,004 images

\- Testing: 7,180 images



\## Baseline Results



ResNet-18, trained from scratch:

\- Epochs: 3

\- Best Validation Accuracy: 84.73%

\- Final Training Accuracy: 87.20%

\- Peak PyTorch GPU Memory: 234.74 MB



These are preliminary validation results, not clinical performance.



\## Project Components



\- core: Training and adaptation engines

\- kernel: Confidence and feature alignment algorithms

\- models: Computer Vision models

\- scripts: Data preparation and training

\- experiments: Research experiments

\- tests: Automated tests



\## Status



Baseline training completed.

Next: GitHub setup, independent evaluation, and TTA development.



\## Disclaimer



Research only. Not for clinical diagnosis.



