from mane.FastChat.fastchat.model.model_adapter import get_generate_stream_function
from FastChat.fastchat.serve.inference import ChatIO, chat_loop
from FastChat.fastchat.utils import str_to_torch_dtype
from FastChat.fastchat.modules.gptq import GptqConfig
from FastChat.fastchat.modules.awq import AWQConfig
from FastChat.fastchat.model.model_adapter import load_model
from FastChat.fastchat.model.model_adapter import get_conversation_template
from FastChat.fastchat.utils import get_context_length

# args Vicuna : 
# Namespace(model_path='/home/chatbot/model_weights/vicuna-7b-v1.5', revision='main', device='cuda', 
#           gpus=None, num_gpus=1, max_gpu_memory=None, dtype=None, load_8bit=False, cpu_offloading=False, 
#           gptq_ckpt=None, gptq_wbits=16, gptq_groupsize=-1, gptq_act_order=False, awq_ckpt=None, awq_wbits=16, 
#           awq_groupsize=-1, enable_exllama=False, exllama_max_seq_len=4096, exllama_gpu_split=None, 
#           enable_xft=False, xft_max_seq_len=4096, xft_dtype=None, conv_template=None, conv_system_msg=None, 
#           temperature=0.7, repetition_penalty=1.0, max_new_tokens=512, no_history=False, style='simple', 
#           multiline=False, mouse=False, judge_sent_end=False, debug=False)


class SimpleChatIO(ChatIO):
    def __init__(self, multiline: bool = False):
        self._multiline = multiline

    def prompt_for_input(self, role) -> str:
        if not self._multiline:
            return input(f"{role}: ")

        prompt_data = []
        line = input(f"{role} [ctrl-d/z on empty line to end]: ")
        while True:
            prompt_data.append(line.strip())
            try:
                line = input()
            except EOFError as e:
                break
        return "\n".join(prompt_data)

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:            
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)
    
    def stream_output2(self, output_stream):
        pre = 0
        for outputs in output_stream:            
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                # print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        # print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)
    
    def print_output(self, text: str):
        print(text)

# model_path='/home/mane/model_weights/vicuna-7b-v1.5'
# revision='main'
# device='cuda'
# gpus=None
# num_gpus=1
# max_gpu_memory=None
# dtype=None
# load_8bit=False
# cpu_offloading=False
# gptq_ckpt=None
# gptq_wbits=16
# gptq_groupsize=-1
# gptq_act_order=False
# awq_ckpt=None
# awq_wbits=16
# awq_groupsize=-1
# enable_exllama=False
# exllama_max_seq_len=4096
# exllama_gpu_split=None
# enable_xft=False
# xft_max_seq_len=4096
# xft_dtype=None
# conv_template=None
# conv_system_msg=None
# temperature=0.7
# repetition_penalty=1.0
# max_new_tokens=2000
# no_history=False
# style='simple'
# multiline=False
# mouse=False
# judge_sent_end=False
# debug=False

# gptq_config=GptqConfig(
#     ckpt=gptq_ckpt or model_path,
#     wbits=gptq_wbits,
#     groupsize=gptq_groupsize,
#     act_order=gptq_act_order,
# )

# awq_config=AWQConfig(
#     ckpt=awq_ckpt or model_path,
#     wbits=awq_wbits,
#     groupsize=awq_groupsize,
# )

# exllama_config=None,
# xft_config=None,

# chatio = SimpleChatIO(multiline)

# model, tokenizer = load_model(
#     model_path,
#     device=device,
#     num_gpus=num_gpus,
#     max_gpu_memory=max_gpu_memory,
#     dtype=dtype,
#     load_8bit=load_8bit,
#     cpu_offloading=cpu_offloading,
#     gptq_config=gptq_config,
#     awq_config=awq_config,
#     exllama_config=False,
#     xft_config=False,
#     revision=revision,
#     debug=debug,
# )

# generate_stream_func = get_generate_stream_function(model, model_path)
# conv = get_conversation_template(model_path)

# # conv.roles[0] = 'USER'
# # conv.roles[1] = 'ASSISTANT'

# user_message = "Peux-tu me donner la liste des opérations pour changer une garniture mécanique sur une machine tournante ? "

# conv.append_message(conv.roles[0], user_message)
# conv.append_message(conv.roles[1], None)

# prompt = conv.get_prompt()

# gen_params = {
#     "model": model_path,
#     "prompt": prompt,
#     "temperature": temperature,
#     "repetition_penalty": repetition_penalty,
#     "max_new_tokens": max_new_tokens,
#     "stop": conv.stop_str,
#     "stop_token_ids": conv.stop_token_ids,
#     "echo": False,
# }

# context_len = get_context_length(model.config)

# output_stream = generate_stream_func(
#     model,
#     tokenizer,
#     gen_params,
#     device,
#     context_len=context_len,
#     judge_sent_end=judge_sent_end,
# )

# # outputs = chatio.stream_output(output_stream)

# outputs2 = chatio.stream_output2(output_stream)
# print ()
# print (outputs2)

