## generate text 
import os, sys
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM, pipeline
import torch

MAX_LEN = 400

prompt = 'This is what I think about life:' ## default prompt 
# prompt = 'prompt.txt' ## default prompt file

model_path = './models/gpt'
# model_path = '../models'

gpt_model = 'EleutherAI/gpt-neo-125M'
# gpt_model = 'EleutherAI/gpt-neo-1.3B'
# gpt_model = 'EleutherAI/gpt-neo-2.7B'
# gpt_model = 'EleutherAI/gpt-j-6B'

#######################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpt_model_file = os.path.join(model_path, gpt_model.split('/')[-1] + ".pt")

print('loading model...')
if os.path.exists(gpt_model_file):
    model = torch.load(gpt_model_file)
else:
    if 'neo' in gpt_model:
        model = AutoModelForCausalLM.from_pretrained(gpt_model)
    else:
        model = GPTJForCausalLM.from_pretrained(gpt_model, revision="float16", torch_dtype=torch.float16)
    torch.save(model, gpt_model_file)
model.to(device)
print('done!')

tokenizer = AutoTokenizer.from_pretrained(gpt_model)

# prompt/response loop...
while True:

    text = input("\n\nINPUT PROMPT TEXT: ")
    text = str(text)
    if len(text) == 0:
        text = prompt ## default text
    if text.endswith('.txt') and os.path.exists(text):
        with open(text, 'r') as f:
            text = f.read().rstrip()
    if len(text) == 0:
        continue

    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    # add the length of the prompt tokens to match with the mesh-tf generation
    max_length = MAX_LEN + ids.shape[1]

    gen_tokens = model.generate(
        ids,
        do_sample=True,
        min_length=max_length,
        max_length=max_length,
        temperature=0.9,
        use_cache=True
        )

    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print("\n\nTEXT GENERATED:")
    print(gen_text)
