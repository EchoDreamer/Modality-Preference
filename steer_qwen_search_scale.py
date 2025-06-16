import json
import argparse
import json,os
from tqdm import tqdm
import copy
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
# from decord import VideoReader, cpu
import torch.nn.functional as F
from PIL import Image
import torch
import h5py

def format_train_coco_image_id(image_id):
    return f"train2014/COCO_train2014_{str(image_id).zfill(12)}.jpg"
def format_val_coco_image_id(image_id):
    return f"val2014/COCO_val2014_{str(image_id).zfill(12)}.jpg"
def data_process(args):
    data = json.load(open(args.question_file, encoding='utf-8'))
    return data

model_predict=None

# default template
instruction_text= " In case there is an inconsistency between the text context and the image content, you should follow the text context rather than the image content. "
instruction_image = " In case there is an inconsistency between the text context and the image content, you should follow the image content rather than the text context. "
QUESTION_TEMPLATE = "{Question} Carefully distinguish between image and text content and output the thinking process in <think> </think> and final answer option in <answer> </answer> tags."
instruction_both=" you must answer the question separately based on text context and the image content and must output two final answers in <answer1> </answer1> and <answer2> </answer2>."
def extract_ASSISTANT(text):
    import re
    match = re.search(r"ASSISTANT:\s*(.*)", text)
    if match:
        assistant_answer = match.group(1)  
        # print(assistant_answer)
        return assistant_answer
    else:
        print("Error!")
        exit()
def extract_assistant_(text):
    import re
    match = re.search(r"assistant\s*(.*)", text)
    if match:
        assistant_answer = match.group(1)  
        return assistant_answer
    else:
        print("Error!")
        exit()

def write_h5(data,name):
    with h5py.File(name, 'w') as f:
        group = f.create_group('data')
        for idx, entry in enumerate(data):
            entry_group = group.create_group(f"entry_{idx}")
            for key in entry.keys():
                # print(key)
                if "hidden_state" in key:
                    entry_group.create_dataset(key, data=entry[key].to('cpu').to(torch.float32).numpy()) 
                else:
                    if key=='context':
                        entry_group.create_dataset(key, data=entry[key]['icc'])
                    else:
                        entry_group.create_dataset(key, data=entry[key])

def model_predict_qwen2vl(question_base,args,image_id,model,processor):
    from qwen_vl_utils import process_vision_info
    if image_id==None:
        image_path=None
    else:
        image_path=os.path.join(args.image_folder, image_id)
    messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": question_base},
                    ],
                }
            ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True
            )
    if image_path is not None:
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        image_inputs, video_inputs = None, None
    inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, do_sample=args.do_sample,max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
    output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
    print(output_text)
    
    return_dict={
        "text":output_text,
    }
    return return_dict

def model_predict_qwen2vl_hidden_state(question_base,args,image_id,model,processor):
    from qwen_vl_utils import process_vision_info
    if image_id==None:
        image_path=None
    else:
        image_path=os.path.join(args.image_folder, image_id)
    messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": question_base},
                    ],
                }
            ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True
            )
    if image_path is not None:
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        image_inputs, video_inputs = None, None
    generation_args = {
    "max_new_tokens": args.max_new_tokens,
    "output_scores": True,
    "output_logits": True,
    "return_dict_in_generate": True,
    "output_hidden_states": True,
    }
    inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
    inputs = inputs.to(model.device)
    outputs=model.generate(**inputs, **generation_args)
    
    generated_ids = outputs.sequences
    hidden_states=outputs.hidden_states

    generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
    output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
    print(output_text)
    tensor_tuple_on_cpu=torch.cat([tensor.cpu() for tensor in hidden_states[0]],dim=0)

    
    return_dict={
        "text":output_text,
        "hidden_states":tensor_tuple_on_cpu[:,-5:,:]
    }
    return return_dict


def open_ended_inference(question_,args):
    if args.mode_type=='base_image' or args.mode_type=='base_text':
        return question_+" Please directly answer the question."
    elif args.mode_type=='icc':
        question_icc1=question_+" Please directly answer the question."
        return question_icc1
    else:
        raise Exception("mode_type should be base_image or base_text or icc")
def multi_answer_qa_inference(question_,gt,hitem,args):
    question_1=question_+"\nA. "+gt+"\nB. "+hitem+"\nAnswer the question by selecting the correct answer A or B."
    question_2=question_+"\nA. "+hitem+"\nB. "+gt+"\nAnswer the question by selecting the correct answer A or B."
    return question_1,question_2
def base_image_inference(temp,model,processor,args,image_id,context,question,num,hidden_tensor):
    result=model_predict(question,args,image_id,model,processor,hidden_tensor)
    temp[f'result{num}']=result["text"]  
    if args.evaluate_type=='output_hidden_states':
        temp[f'result_hidden_states_{num}']=result['hidden_states']
    return temp
def base_text_inference(temp,model,processor,args,image_id,context,question,num,hidden_tensor):
    question_="Text context: "+context+"\n"+question
    result=model_predict(question_,args,None,model,processor)
    temp[f'result{num}']=result["text"] 
    if args.evaluate_type=='output_hidden_states':
        temp[f'result_hidden_states_{num}']=result['hidden_states']
    return temp

def both_multi_no_specific_inference(temp,model,processor,args,image_id,context,question,num,hidden_tensor):
    question_="Text context: "+context+"\n"+question
    result=model_predict(question_,args,image_id,model,processor)
    temp[f'result{num}']=result["text"]  
    if args.evaluate_type=='output_hidden_states':
        temp[f'result_hidden_states_{num}']=result['hidden_states']
    return temp

def icc_inference(temp_icc,model,processor,args,image_id,context,question,num,hidden_tensor):

    qestion_icc_text="Text context: "+context+"\n"+instruction_text+"\n"+question
    qestion_icc_image="Text context: "+context+"\n"+instruction_image+"\n"+question

    icc_result_image=model_predict(qestion_icc_image,args,image_id,model,processor)
    icc_result_text=model_predict(qestion_icc_text,args,image_id,model,processor)
    temp_icc[f'result_image_{num}']=icc_result_image['text']
    
    
    temp_icc[f'result_text_{num}']=icc_result_text['text']
    if args.evaluate_type=='output_hidden_states':
        temp_icc[f'result_image_hidden_states_{num}']=icc_result_image['hidden_states']
        temp_icc[f'result_text_hidden_states_{num}']=icc_result_text['hidden_states']
    return temp_icc
def inference(questions,args,image_train_list,ans_file,model,processor):

    context_name="icc"
    
    if args.mode_type=='base_image':
        inference_function=base_image_inference
    elif args.mode_type=='base_text':
        inference_function=base_text_inference
    elif args.mode_type=='icc':
        inference_function=icc_inference
    elif args.mode_type=='single':
        inference_function=both_multi_no_specific_inference
    else:
        raise Exception("mode_type should be base_image or base_text or icc")
    if args.evaluate_type=='output_hidden_states':
        with h5py.File(args.hidden_states_answer_file, 'w') as f:
            group = f.create_group('data')
            idx=0
            for sample in tqdm(questions):
                try:
                    context=sample['context'][context_name]
                except:
                    context=None
                if "COCO" in sample['image_id']:
                    image_id=sample['image_id']
                else:
                    image_id=format_train_coco_image_id(sample['image_id']) if sample['image_id'] in image_train_list else format_val_coco_image_id(sample['image_id'])
                temp=copy.deepcopy(sample)
                if "open" in args.inference_type:
                    question_1=open_ended_inference(sample['question'],args)
                    temp['question_now']=question_1
                    temp=inference_function(temp,model,processor,args,image_id,context,question_1)
                    ans_file.write(json.dumps(temp) + "\n")
                else:
                    question_1,question_2=multi_answer_qa_inference(sample['question'],sample['gt'],sample['hitem'],args)
                    temp['question_now1']=question_1
                    temp['question_now2']=question_2
                    hidden_tensor=None
                    temp=inference_function(temp,model,processor,args,image_id,context,question_1,1,hidden_tensor)
                    temp=inference_function(temp,model,processor,args,image_id,context,question_2,2,hidden_tensor)
                temp_=copy.deepcopy(temp)
                entry_group = group.create_group(f"entry_{idx}")
                keys_del=[]
                for key in temp.keys():
                    if "hidden_states" in key:
                        keys_del.append(key)
                for key in keys_del:
                    del temp[key]
                ans_file.write(json.dumps(temp) + "\n")
                for key in temp_.keys():
                    if "hidden_state" in key:
                        entry_group.create_dataset(key, data=temp_[key].to('cpu').to(torch.float32).numpy()) 
                    else:
                        if key=='context':
                            entry_group.create_dataset(key, data=temp_[key]['icc'])
                        else:
                            entry_group.create_dataset(key, data=temp_[key])
                idx+=1
                # torch.save(final_hidden_states_list, args.hidden_states_answer_file)
    else:
        for sample in tqdm(questions):
            try:
                context=sample['context'][context_name]
            except:
                context=None
            if "COCO" in sample['image_id']:
                image_id=sample['image_id']
            else:
                image_id=format_train_coco_image_id(sample['image_id']) if sample['image_id'] in image_train_list else format_val_coco_image_id(sample['image_id'])
            temp=copy.deepcopy(sample)
            if "open" in args.inference_type:
                question_1=open_ended_inference(sample['question'],args)
                temp['question_now']=question_1
                temp=inference_function(temp,model,processor,args,image_id,context,question_1)
                ans_file.write(json.dumps(temp) + "\n")
            else:
                question_1,question_2=multi_answer_qa_inference(sample['question'],sample['gt'],sample['hitem'],args)
                temp['question_now1']=question_1
                temp['question_now2']=question_2
                hidden_tensor=None
                temp=inference_function(temp,model,processor,args,image_id,context,question_1,1,hidden_tensor)
                temp=inference_function(temp,model,processor,args,image_id,context,question_2,2,hidden_tensor)
            ans_file.write(json.dumps(temp) + "\n")

def write_h5(data,name):
    with h5py.File(name, 'w') as f:
        group = f.create_group('data')

        for idx, entry in enumerate(data):

            entry_group = group.create_group(f"entry_{idx}")
            for key in entry.keys():
                if "hidden_state" in key:
                    entry_group.create_dataset(key, data=entry[key].to('cpu').to(torch.float32).numpy()) 
                else:
                    if key=='context':
                        entry_group.create_dataset(key, data=entry[key]['icc'])
                    else:
                        entry_group.create_dataset(key, data=entry[key])
         
def load_model_qwen2vl(args):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,Qwen2VLForConditionalGeneration
    from peft import PeftModel, LoraConfig, TaskType,PeftConfig
    if "2.5" in args.model_path:
        QweVLForConditionalGeneration=Qwen2_5_VLForConditionalGeneration
    else:
        QweVLForConditionalGeneration=Qwen2VLForConditionalGeneration
    
    if "qvq" in args.model_path or "72" in args.model_path:
        model = QweVLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto",attn_implementation='flash_attention_2')
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = QweVLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map=None,attn_implementation='flash_attention_2')
        model = model.to(device)

    if args.lora_path is not None:
        lora_config = PeftConfig.from_pretrained(args.lora_path)
        lora_config.inference_mode = True
        model = PeftModel.from_pretrained(model, model_id = args.lora_path,config=lora_config)  

    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    global model_predict
    if args.evaluate_type=='output_hidden_states':
        model_predict=model_predict_qwen2vl_hidden_state
    else:
        model_predict=model_predict_qwen2vl
    return model,processor     


def template_define(args):
    global instruction_text,instruction_image,instruction_both,QUESTION_TEMPLATE

    if args.inference_type=="single_multi_no_specific":
        QUESTION_TEMPLATE = "{Question}\nAnswer the question by selecting the correct answer A or B."
    else:
        instruction_image = "In case there is an inconsistency between the text context and the image content, you should follow the image content rather than the text context. "
        QUESTION_TEMPLATE = "{Question} Carefully distinguish between image and text content and output the thinking process in <think> </think> and final answer option in <answer> </answer> tags."   
        instruction_both="you must answer the question separately based on text context and the image content and must output two final answers in <answer1> </answer1> and <answer2> </answer2>."
        if args.text_template=='ori':
            instruction_text= "In case there is an inconsistency between the text context and the image content, you should follow the text context rather than the image content. "
        elif args.text_template=='template1':
            instruction_text = "In case there is an inconsistency between the Text Context and the input image content, you must rely on the text context rather than the input image content. "
        else:
            raise Exception("text_template should be ori or template1")
def str_to_bool(value):
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected. Use 'True' or 'False'.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="../huggingface/model/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="../huggingface/dataset/coco2014/")
    parser.add_argument("--question-file", type=str, default="data/final_data/annotate_final.json")
    parser.add_argument("--answer-file", type=str, default="results/v6_steer_single_test/icc/base_multi/Qwen2.5-VL-7B-Instruct.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--do_sample",type=str_to_bool,default=False)
    parser.add_argument("--mode-type", type=str, default="icc",help="base_image, base_text, icc, both, single")  
    parser.add_argument("--person-keypoints-train2014", type=str, default="../huggingface/dataset/coco2014/annotations/person_keypoints_train2014.json")
    parser.add_argument("--person-keypoints-val2014", type=str, default="../huggingface/dataset/coco2014/annotations/person_keypoints_val2014.json")
    parser.add_argument("--special-text", type=str, default="icc")
    parser.add_argument("--inference-type", type=str, default="base_multi",help="base_multi, single_multi_no_specific")  
    parser.add_argument("--debug_", type=str_to_bool, default=True,help="True: sample 10 data")
    parser.add_argument("--evaluate-type", type=str, default="steer",help="predict_qa, predict_logit, output_hidden_states and steer") 
    parser.add_argument("--hidden-states-answer-file", type=str, default="results/version_multi_answer_text_context_first/base/qvq-conflict-base.h5",help="hidden_states file path")
    parser.add_argument("--metric-normalize", type=str, default="L2")
    parser.add_argument("--text_template", type=str, default="ori",help="ori, template1")
    args = parser.parse_args()
    questions=data_process(args)
    if args.debug_:
        questions=questions[:3]
    print(len(questions))
    image_train=json.load(open(args.person_keypoints_train2014, encoding='utf-8'))
    image_train_list=[str(item['id']).zfill(12) for item in image_train['images']]
    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)
    ans_file = open(args.answer_file, "w")
    if "Qwen2" in args.model_path:
        model,processor=load_model_qwen2vl(args)
    else:
        print("Model loader Error!")
        exit()
    template_define(args)
    inference(questions,args,image_train_list,ans_file,model,processor)
    ans_file.close()