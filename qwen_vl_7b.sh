#!/bin/bash                         
module load amd/Anaconda/2023.3
source activate qwenvl
# Add a new key named 'icc' in the "context" of evaluat_final.json, and move the original content of 'context' into the 'icc' dictionary.
export CUDA_VISIBLE_DEVICES=$1
model=Qwen2.5-VL-7B-Instruct
text_template=template1
version=hidden_states_single_$text_template
max_new_tokens=512 
model_path=../huggingface/model/$model
# model type: 
# single: input both text and vision context but no instruction towards any modality
# base_text: only text context as input 
# base_image: only vision context as input
# icc: input both text and vision context with instruction towards vision or text modality
mode_type=$2
# single_multi_no_specific
# base_multi
inference_type=$3
evaluate_type=output_hidden_states
debug_=False
# using text context with grammer error by replacing the query file
question_path=data/evaluate_final.json
answer_path=results/$version/$mode_type/$inference_type/$model.jsonl
# using noisy images by replacing the image folder
image_folder=../huggingface/dataset/coco2014/
hidden_states_answer_file=results/$version/$mode_type/$inference_type/$model.h5


python steer_qwen_search_scale.py \
    --model-path $model_path \
    --question-file $question_path \
    --answer-file $answer_path \
    --image-folder $image_folder \
    --inference-type $inference_type \
    --mode-type $mode_type \
    --debug_ $debug_ \
    --max-new-tokens $max_new_tokens \
    --hidden-states-answer-file $hidden_states_answer_file \
    --evaluate-type $evaluate_type \
    --text_template $text_template 