# LFX-Mentorship-Proposal-3169

This repository is about the LFX Mentorship (Mar-May, 2024): [Integrate Intel Extension for Transformers as a new WASI-NN backend #3169](https://github.com/WasmEdge/WasmEdge/issues/3169).

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [1. Intel Extension For Transformers](#1-intel-extension-for-transformers)
   * [Install dependency](#install-dependency)
   * [Download model](#download-model)
   * [Run Example](#run-example)
   * [Troubleshooting](#troubleshooting)
- [2 Chat Example](#2-chat-example)
   * [Install WasmEdge](#install-wasmedge)
   * [Download model and Excute](#download-model-and-excute)
   * [Result](#result)

<!-- TOC end -->

<!-- TOC --><a name="1-intel-extension-for-transformers"></a>
# 1. Intel Extension For Transformers
<!-- TOC --><a name="install-dependency"></a>
## Install dependency
Although the intel extension for transformers readme only installs intel-extension-for-transformers, I have encountered some problems with the not-found module. I installed the remaining dependencies based on the error message from running the example code.

``` shell
pip install accelerate
pip install datasets
pip install intel_extension_for_pytorch
pip install intel-extension-for-transformers
git clone git@github.com:intel/neural-speed.git
cd neural-speed
pip install -r requirements.txt
pip install .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

<!-- TOC --><a name="download-model"></a>
## Download model
Using git lfs to download model from huggingface.
``` shell
wget https://github.com/git-lfs/git-lfs/releases/download/v3.4.1/git-lfs-linux-amd64-v3.4.1.tar.gz
tar zxvf git-lfs-linux-amd64-v3.4.1.tar.gz
cd git-lfs-3.4.1/
sudo ./install
git lfs install
git clone https://huggingface.co/Intel/neural-chat-7b-v3-1
```

<!-- TOC --><a name="run-example"></a>
## Run Example
I modify the example code from the repository to let it show the readable output.

``` python
from transformers import AutoTokenizer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
model_name = "neural-chat-7b-v3-1"     
prompt = "Once upon a time, there existed a little girl,"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
outputs = model.generate(inputs)


decode_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decode_output)
```

Run the python script, then you can see the following output.

```
Once upon a time, there existed a little girl, who was born with a gift. She could see the world in a different way. She could see the world through the eyes of a child. She could see the world as it was, and as it could be.

She could see the world as it was, and she saw that it was filled with beauty and wonder. She saw that there were people who loved and cared for one another. She saw that there were people who were kind and generous. She saw that there were people who were brave and strong.

She could see the world as it could be, and she saw that it could be filled with even more beauty and wonder. She saw that people could learn to love and care for one another even more deeply. She saw that people could become kinder and more generous. She saw that people could become braver and stronger.

The little girl knew that she had a special gift, and she wanted to share it with the world. She wanted to help people see the world through her eyes, so that they could experience the beauty and wonder that she saw. She wanted to inspire people to become better versions of themselves, to love more, to care more, to be kinder, more generous, braver, and stronger.

So, the little girl began to share her gift with others. She started by telling her stories to her friends and family. She shared her stories with her teachers and her classmates. She even shared her stories with people she met in her travels.

As she shared her stories, people began to listen. They began to see the world through her eyes, and they started to believe in the possibility of a better world. They began to change their own perspectives and actions. They started to love more, to care more, to be kinder, more generous, braver, and stronger.

The little girl's gift had a ripple effect. As more and more people saw the world through her eyes, they inspired others to do the same. The world began to change, becoming more beautiful and filled with wonder. People learned to love and care for one another more deeply. They became kinder, more generous, braver, and stronger.

And so, the little girl's gift of seeing the world through the eyes of a child transformed the world into a more loving, caring, and inspiring place. The world became a better place for everyone, thanks to the little girl's special gift.
```

<!-- TOC --><a name="troubleshooting"></a>
## Troubleshooting
This error may caused by not correctly installing torchvision. Reinstalling torch and torchvision can solve it.

``` shell
ValueError: Could not find the operator torchvision::nms. Please make sure you have already registered the operator and (if registered from C++) loaded it via torch.ops.load_library.
```

This error shows your model path is not correctly set.

``` shell
python: can't open file 'Transformers-based_extension_APIs.py': [Errno 2] No such file or directory
```

<!-- TOC --><a name="2-chat-example"></a>
# 2 Chat Example

<!-- TOC --><a name="install-wasmedge"></a>
## Install WasmEdge

``` shell
git clone git@github.com:WasmEdge/WasmEdge.git
cd WasmEdge
git checkout hydai/0.13.5_ggml_lts

cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release \
  -DWASMEDGE_PLUGIN_WASI_NN_BACKEND="GGML" \
  -DWASMEDGE_PLUGIN_WASI_NN_GGML_LLAMA_BLAS=OFF \
  .

cmake --build build

# For the WASI-NN plugin, you should install this project.
sudo cmake --install build

```

The detailed log message in [here](build_wasm_log.txt)

<!-- TOC --><a name="download-model-and-excute"></a>
## Download model and Excute

``` shell
cd llama
cargo build --target wasm32-wasi --release
cp target/wasm32-wasi/release/wasmedge-ggml-llama.wasm ./wasmedge-ggml-llama.wasm

curl -LO https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf

wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf \
  wasmedge-ggml-llama.wasm default

```

<!-- TOC --><a name="result"></a>
## Result

The following image is the result of interacting with the chatbot.

![chat-example.png](image%2Fchat-example.png)