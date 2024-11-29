# DKP-ADS：Domain Knowledge Prompt-driven Large Model with Few Labeling Data via Multi-task Learning for Assessment of Staple Crop Disease Severity

## paper

## Abstract

## Environment

- **Set up our environment**

  ```bash
  conda create -n your_env_name python=x.x
  conda activate your_env_name
  pip install -r reqirements.txt
  pip install -e .
  ```

## Training

- **Step1** Fine-tune GroundingDINO with your own datasets.

  ```bash
  cd /Grounding-Dino-FineTuning
  python train.py
  ```

- **Step2** Use fine-tuned GroundingDINO weights to implement DKB-ADS.

  ```bash
  cd /main
  python main.py
  ```

## Validation

- **Run the code of validation:**

  ```bash
  cd /main
  python val.py
  ```

## Predict

- **Run the code of testing:**

  ```bash
  cd /main
  python predict.py
  ```
