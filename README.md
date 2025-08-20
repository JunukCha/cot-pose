## CoT-Pose: Chain-of-Thought Reasoning for 3D Pose Generation from Abstract Prompts (ICCVW'25)

![pipeline](https://github.com/user-attachments/assets/bc649325-aac3-4fd0-b4ee-ce7b11f3b8bf)

[Paper(Arxiv)](https://arxiv.org/abs/2508.07540)

## 1. Installation

Install all required dependencies with:

```bash
source scripts/install.sh
```

---

## 2. Download Checkpoints

### 2.1 PoseScript Checkpoints
```bash
source scripts/download_posescript_ckpts.sh
```

### 2.2 UniPose Checkpoints
```bash
source scripts/download_unipose_ckpts.sh
```

### 2.3 CoT-Pose Checkpoints
```bash
source scripts/download_cot_pose_ckpts.sh
```

---

## 3. Training

```bash
source scripts/train.sh
```

---

## 4. Testing

```bash
source scripts/test.sh
```

---

## 5. Demo

```bash
source scripts/demo.sh
```
