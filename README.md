
<div align="center">

# Evaluating and Steering Modality Preferences in Multimodal Large Language Model



<div>
    <a href='https://arxiv.org/abs/2505.20977' target='_blank'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<!--     <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a> -->
</div>

</div>

## Project Overview
Multimodal large language models (MLLMs) have achieved remarkable performance on complex tasks with multimodal context. However, it is still understudied whether they exhibit modality preference when processing multimodal contexts. To study this question, we first build a $MC^2$ benchmark under controlled evidence conflict scenarios to systematically evaluate modality preference, which is the tendency to favor one modality over another when making decisions based on multimodal conflicting evidence. Our extensive evaluation reveals that all 18 tested MLLMs generally demonstrate clear modality bias, and modality preference can be influenced by external interventions. An in-depth analysis reveals that the preference direction can be captured within the latent representations of MLLMs. Built on this, we propose a probing and steering method based on representation engineering to explicitly control modality preference without additional fine-tuning or carefully crafted prompts. Our method effectively amplifies modality preference toward a desired direction and applies to downstream tasks such as hallucination mitigation and multimodal machine translation, yielding promising improvements.


## Install 🛠️
Download and install the specific Transformers from our Repository [**🤗Huggingface**](https://huggingface.co/271754echo/MC2)
```bash
unzip transformers-main.zip
pip install -r requirements.list
cd transformers-main
pip install -e . 
```

## Data
We provide the data in $MC^2$ for evaluating modality preference and controlling modality preference through noisy images or text context with grammer errors in this repository.

The complete data of $MC^2$ can be found in [**🤗Huggingface**](https://huggingface.co/271754echo/MC2)

Add a new key named 'icc' in the "context" of evaluat_final.json, and move the original content of 'context' into the 'icc' dictionary.

## Code 
### Setting
**1. model type**: including single, base_text, base_image and icc

single: input both text and vision context but no instruction towards any modality

base_text: only text context as input 

base_image: only vision context as input

icc: input both text and vision context with instruction towards vision or text modality

**2. inference_type**: including single_multi_no_specific and base_multi

single_multi_no_specific: for single in model type

base_multi: for others in model type

### Inference:
```bash
bash qwen_vl_7b.sh 0 icc base_multi
bash qwen_vl_7b.sh 0 single single_multi_no_specific
bash qwen_vl_7b.sh 0 base_text base_multi
bash qwen_vl_7b.sh 0 base_image base_multi
```
Saving hidden_states in results/$version/$mode_type/$inference_type/$model.h5

### PCA analyse
```bash
python pca.py
```
- For pca.py, set the file paths for `file_single` and `file_instruction` as follows:

  - To generate the hidden states for `file_single`, run:
    ```bash
    bash qwen_vl_7b.sh 0 single single_multi_no_specific
    ```
  - To generate the hidden states for `file_instruction`, run:
    ```bash
    bash qwen_vl_7b.sh 0 icc base_multi
    ```

