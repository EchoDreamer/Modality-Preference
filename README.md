
<div align="center">

# <img src="left-right_brain.png" height="28px"> Evaluating and Steering Modality Preferences in Multimodal Large Language Model



<div>
    <a href='https://arxiv.org/abs/2505.20977' target='_blank'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<!--     <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a> -->
</div>

</div>

## Project Overview
Multimodal large language models (MLLMs) have achieved remarkable performance on complex tasks with multimodal context. However, it is still understudied whether they exhibit modality preference when processing multimodal contexts. To study this question, we first build a $MC^2$ benchmark under controlled evidence conflict scenarios to systematically evaluate modality preference, which is the tendency to favor one modality over another when making decisions based on multimodal conflicting evidence. Our extensive evaluation reveals that all 18 tested MLLMs generally demonstrate clear modality bias, and modality preference can be influenced by external interventions. An in-depth analysis reveals that the preference direction can be captured within the latent representations of MLLMs. Built on this, we propose a probing and steering method based on representation engineering to explicitly control modality preference without additional fine-tuning or carefully crafted prompts. Our method effectively amplifies modality preference toward a desired direction and applies to downstream tasks such as hallucination mitigation and multimodal machine translation, yielding promising improvements.


## Code

We provide the data in $MC^2$ for evaluating modality preference and controlling modality preference through noisy images or text context with grammer errors in this repository.

The complete data of $MC^2$ can be found in [**🤗Huggingface**](https://huggingface.co/271754echo/MC2)


