# Update

## We revise the caption of Figure2-4 and Figure 6 by incorporating more detailed experimental settings and concise analytical summaries.

## Upload the new steering prompt in the prompt.py

## We add the mechanistic analysis of modality preference evolution via the Logit Lens in Text_Preference.png and Vision_Preference.png.
1. Method: We project the layer-wise hidden states from the residual stream of Qwen2.5VL-7B into the unembedding matrix under modal conflict prompts. The green and purple lines represent the average logit scores for the Preference Modality and Competitor Modality, respectively. The blue line denotes the Modality Contrastive Margin (MPM), which quantifies the preference gap. 

2. Key Observations: (1) In layers 0–19, the MPM remains near zero, indicating a stage of raw feature extraction without modal bias.(2) Modality preference first surfaces at layers 20–23, identifying this region as the critical decision point for preference arbitration.(3) While the margin expands in later layers (24+), these layers represent the refinement of a pre-established decision. This mechanistic transition validates why $L_{20–23}$ serves as the most effective intervention window for our steering method.

