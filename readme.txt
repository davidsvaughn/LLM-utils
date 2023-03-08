============================================================================================================
# LOCAL SETUP

virtualenv -p python3.8 venv
source venv/bin/activate

python -m pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0+cu115 -f https://download.pytorch.org/whl/cu115/torch_stable.html

python -m pip install transformers==4.25.1
python -m pip install datasets accelerate triton==1.0.0
python -m pip install evaluate scikit-learn scipy

# git clone https://github.com/davidsvaughn/DeepSpeed.git && cd DeepSpeed
# DS_BUILD_OPS=1 pip install .
# cd ..


## fine-tune LLM for score prediction
python gpt_train.py configs/train_roberta.json


## evaluate LLM for score prediction
python gpt_train.py configs/eval_roberta.json



## generate text
python gpt_gen.py




============================================================================================================
# AWS SETUP

------------------
## AWS : dsvJ4x ##
------------------

Deep Learning AMI GPU PyTorch 1.13.0 (Ubuntu 20.04)
* NVIDIA driver version: 515.65.01
* CUDA version: 11.7
* Python 3.9.13


sudo apt install python3-virtualenv
virtualenv -p python3.8 venv
source venv/bin/activate

pip install torch torchvision torchaudio
pip install datasets transformers accelerate triton==1.0.0

pip install evaluate scikit-learn scipy

[ sudo apt-get update && sudo apt-get install -y libaio-dev ??? ]

git clone https://github.com/davidsvaughn/DeepSpeed.git
cd DeepSpeed
DS_BUILD_OPS=1 pip install .
cd ..

ds_report
DeepSpeed general environment info:
torch install path ............... ['/home/ubuntu/venv/lib/python3.8/site-packages/torch']
torch version .................... 1.13.0+cu117
torch cuda version ............... 11.7
torch hip version ................ None
nvcc version ..................... 11.7
deepspeed install path ........... ['/home/ubuntu/venv/lib/python3.8/site-packages/deepspeed']
deepspeed info ................... 0.7.7+5c451eb, 5c451eb, master
deepspeed wheel compiled w. ...... torch 1.13, cuda 11.7


============================================================================================================

ssh -i ai-training-oregon.pem ubuntu@34.217.112.200

rsync -azP -e "ssh -i ai-training-oregon.pem" /home/david/code/davidsvaughn/conv/gptj/Finetune_GPTNEO_GPTJ6B/finetuning_repo ubuntu@34.217.112.200:/home/ubuntu/Finetune_GPTNEO_GPTJ6B

rsync -azP -e "ssh -i ai-training-oregon.pem" gpt_gen.py ubuntu@34.217.112.200:/home/ubuntu/Finetune_GPTNEO_GPTJ6B/finetuning_repo