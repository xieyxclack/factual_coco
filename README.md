# Factual CoCo: A metric for factual consistency in text summarization via counterfactual estimation

The implementation of *Factual Consistency Evaluation for Text Summarization via Counterfactual Estimation* in PyTorch.
<br/> <br/>

## Requirements
* Python version 3.6 
* Torch version 1.6.0
* spaCy v3.1 (Install from [here](https://spacy.io/usage)) 

Besides, you need to download the model used in spaCy for part-of-speech (pos) tagging
```
python -m spacy download en_core_web_sm
```
or download from &nbsp;*en_core_web_sm-3.1.0-py3-none-any.whl*&nbsp; from [here](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.1.0), and then run `pip install en_core_web_sm-3.1.0-py3-none-any.whl`.
<br/> <br/>


## Quick Start
### 1. Clone the source code
```
git clone http://gitlab.alibaba-inc.com/yuexiang.xyx/factual_coco.git
```

### 2. Install fairseq
We provide [BART](https://arxiv.org/abs/1910.13461) as the scoring model adopted in CoCo, and implement it via [fairseq](https://github.com/pytorch/fairseq) (which is provided in this repository). And you can install fairseq via:
```
cd factual_coco
pip install --editable ./
```
If you would like to adopt other summarization model as the scoring model, you can skip this step and implement your own scoring model.


### 3. Provide model path and data path
Before execute `run_coco.py` to get the coco score, you should provide:
* `model_path`: The path to the scoring model, which is an independent summarization model, and it is not necessary to be the model that generates the evaluated summaries.
* `data_path`: The path to the source documents (named as source.txt) and summaries (named as summary.txt). One document/summary per line. (We provide an example in the `data` folder)
<br/>

**Note**: You might need to modify the `load_model` function in the code according scoring model you use.  
In this repository, we adopt [BART](https://arxiv.org/abs/1910.13461) as the scoring model and implement it via [fairseq](https://github.com/pytorch/fairseq).
The checkpoints can be downloaded from [here](https://github.com/pytorch/fairseq/tree/master/examples/bart), including &nbsp;*bart.large.cnn*&nbsp; and &nbsp;*bart.large.xsum*&nbsp;.  
Take &nbsp;*bart.large.cnn*&nbsp; as example, the model path should include:
```
bart.large.cnn
│   model.pt
│   dict.txt (it can be a copy of the dict.source.txt or dict.target.txt)
```

### 4. Get CoCo scores
```
python3 run_coco.py --model_path /path/to/model --data_path /path/to/data --output_file coco_score.txt --mask token
```
* `mask` is used to set up the mask strategy (one of ['token', 'span', 'sent', 'doc'], more details can be found in the paper). And you can design your own mask strategies in the mask function.
* `output_file` denotes the file to save the generated coco scores. 


<br/> <br/>
## Cite
If you find this repository useful for your research or development, please cite the following [paper](https://arxiv.org/abs/2108.13134):
```
@article{xie2021factual,
  title={Factual Consistency Evaluation for Text Summarization via Counterfactual Estimation},
  author={Xie, Yuexiang and Sun, Fei and Deng, Yang and Li, Yaliang and Ding, Bolin},
  journal={arXiv preprint arXiv:2108.13134},
  year={2021}
}
```

