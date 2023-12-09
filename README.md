# :sweat_drops: [Linguistic-Based Watermarking for Text Authentication](https://arxiv.org/abs/2305.08883)

Official implementation of the watermark injection and detection algorithms presented in the [paper](https://arxiv.org/abs/2305.08883):

"Linguistic-Based Watermarking for Text Authentication" by _Xi Yang, Kejiang Chen, Weiming Zhang, Chang Liu, Yuang Qi, Jie Zhang, Han Fang, and Nenghai Yu_.  

## Requirements
- Python 3.9
- check requirements.txt
```sh
pip install -r requirements.txt
pip install git+https://github.com/JunnYu/WoBERT_pytorch.git  # Chinese word-level BERT model
python -m spacy download en_core_web_sm
```
- For Chinese, please download the [pre-trained Chinese word vectors](https://drive.google.com/file/d/1Zh9ZCEu8_eSQ-qkYVQufQDNKPC4mtEKR/view) and place it in the root directory.

## Repo contents

The watermark injection and detection modules are located in the `models` directory. `watermark_original.py` implements the iterative algorithms as described in the paper. `watermark_faster.py` introduces batch processing to speed up the watermark injection algorithm and the precise detection algorithm.

We provide two demonstrations, `demo_CLI.py` and `demo_gradio.py`, which correspond to command-line interaction and graphical interface interaction respectively.

## Demo Usage
> Click on the GIFs to enlarge them for a better experience.
### Graphical User Interface
```sh
$ python demo_gradio.py --language English --tau_word 0.8 --lamda 0.83
```
<p align="center">
  <img src="images/en_gradio.gif" />
</p>

```sh
$ python demo_gradio.py --language Chinese --tau_word 0.75 --lamda 0.83
```
<p align="center">
  <img src="images/cn_gradio.gif" />
</p>

### Command Line Interface
```sh
$ python demo_CLI.py --language English --tau_word 0.8 --lamda 0.83
```
<p align="center">
  <img src="images/eng_cli.gif" />
</p>

```sh
$ python demo_CLI.py --language Chinese --tau_word 0.75 --lamda 0.83
```

<p align="center">
  <img src="images/cn_cli.gif" />
</p>


