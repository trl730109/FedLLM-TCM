[**中文**](./README.md) | [**English**](./README_EN.md)

<p align="center">
    <br>
    <img src="https://github.com/michael-wzhu/ChatMed/blob/main/pics/ChatMed.png" width="355"/>
    <br>
</p>
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-LLaMA-Alpaca.svg?color=blue&style=flat-square">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ymcui/Chinese-LLaMA-Alpaca">
</p>

2023 HKUST(GZ) MPhil program: Towards Smart TCM 1. This project is to build a smart TCM chatbot with FedLLM fine-tuning based on Llama2-7b.
Modified from the following projects
- 🚀 [ChatMed-Consult](https://github.com/michael-wzhu/ChatMed) : 基于[中文医疗在线问诊数据集ChatMed_Consult_Dataset](https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset)的50w+在线问诊+ChatGPT回复作为训练集。模型主干为[LlaMA-7b](https://github.com/facebookresearch/llama),融合了[Chinese-LlaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)的LoRA权重与中文扩展词表，然后再进行基于LoRA的参数高效微调。我们将全部数据和代码都进行了公开。我们也将部署一个在线Gradio demo, 敬请关注。
- 🚀 [ShenNong-TCM-LLM](https://github.com/michael-wzhu/ShenNong-TCM-LLM) : 大模型赋能中医药传承。这一模型的训练数据为[中医药指令数据集ChatMed_TCM_Dataset](https://huggingface.co/datasets/michaelwzhu/ShenNong_TCM_Dataset)。以我们开源的[中医药知识图谱](https://github.com/ywjawmw/TCM_KG)为基础，采用以实体为中心的自指令方法(entity-centric self-instruct)，调用ChatGPT得到11w+的围绕中医药的指令数据。ShenNong-TCM-LLM模型也是以LlaMA为底座，采用LoRA微调得到。


----

### 快速上手

在使用[ChatMed-Consult](https://huggingface.co/michaelwzhu/ChatMed-Consult)之前，大家需要准备好LlaMA-7b底座模型，详细操作见[LlaMA-7b模型准备](https://github.com/michael-wzhu/ChatMed/blob/main/src/chatmed_llama_peft/LlaMA-7b%E6%A8%A1%E5%9E%8B%E5%87%86%E5%A4%87.md)。

LlaMA-7b底座模型准备好后，下载[ChatMed-Consult的LoRA权重](https://huggingface.co/michaelwzhu/ChatMed-Consult)，在3090显卡(或者更强的显卡) 运行以下命令，启动一个简单的基于flask的web service:

```bash
python src/web_services/web_service_simple.py
```

然后运行 
```bash
python src/web_services/web_service_test.py
```

上面的脚本主要是运行了test_examples.json文件中提供了测试用例。在使用自己的测试用例时，请注意保持格式一致。

### Fine-tune

首先，大家需要准备好LlaMA-7b底座模型，保存于`resources/chinese-llama-alpaca-plus-lora-7b`路径。数据集采用[中医药指令数据集ChatMed_TCM_Dataset](https://huggingface.co/datasets/michaelwzhu/ShenNong_TCM_Dataset)。我们采用deepspeed实现分布式训练：

```bash
./src/chatmed_llama_peft/run_train.sh
```

训练脚本中使用的是8张显卡，大家根据自己的服务器情况调整超参数。








