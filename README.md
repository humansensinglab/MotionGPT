# MotionGPT

The official PyTorch implementation of the paper [**"MotionGPT: Human Motion Synthesis with Improved Diversity and Realism via GPT-3 Prompting"**](http://humansensing.cs.cmu.edu/sites/default/files/MotionGPT%20Human%20Motion%20Synthesis.pdf).


#### Bibtex
If you find this code useful in your research, please cite:

```
@inproceedings{ribeiro2024motiongpt,
  title={MotionGPT: Human Motion Synthesis with Improved Diversity and Realism via GPT-3 Prompting},
  author={Ribeiro-Gomes, Jose and Cai, Tianhui and Milacski, Zolt{\'a}n A and Wu, Chen and Prakash, Aayush and Takagi, Shingo and Aubel, Amaury and Kim, Daeil and Bernardino, Alexandre and De La Torre, Fernando},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5070--5080},
  year={2024}
}
```

### NOTE: WIP

This code currently only has the instructions for inference. Training and data preparation will come shortly.

## Getting started

This code was tested on `Ubuntu 18.04 LTS` and requires:

* Python 3.7
* conda3 or miniconda3
* CUDA capable GPU (tested on NVidia RTX A4000 16GB)

### 1. Setup environment

Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```


Setup conda env:
```shell
conda env create -f environment.yml
conda activate motiongpt
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
pip install sentence_transformers 
```

Download dependencies:

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```



<!-- ### 2. Get data

<details>
  <summary><b>Text to Motion</b></summary>

There are two paths to get the data:

(a) **Go the easy way if** you just want to generate text-to-motion (excluding editing which does require motion capture data)

(b) **Get full data** to train and evaluate the model.


#### a. The easy way (text only)

**HumanML3D** - Clone HumanML3D, then copy the data dir to our repository:

```shell
cd ..
git clone https://github.com/EricGuo5513/HumanML3D.git
unzip ./HumanML3D/HumanML3D/texts.zip -d ./HumanML3D/HumanML3D/
cp -r HumanML3D/HumanML3D motion-diffusion-model/dataset/HumanML3D
cd motion-diffusion-model
```


#### b. Full data (text + motion capture)

**HumanML3D** - Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git),
then copy the result dataset to our repository:

```shell
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```

**KIT** - Download from [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git) (no processing needed this time) and the place result in `./dataset/KIT-ML`
</details> -->


### 2. Download the pretrained models

Download the model(s) you wish to use, then unzip and place them in `./save/`. 

[link](https://drive.google.com/drive/folders/1widBXdRfuoNLv9CJEmW0GURSaw59cskQ?usp=sharing)




## Motion Synthesis
<!-- ### Generate from test set prompts

```shell
python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --num_samples 10 --num_repetitions 3
```

### Generate from your text file

```shell
python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --input_text ./assets/example_text_prompts.txt
``` -->

### Generate a single prompt

```shell
python -m sample.generate --model_path ./save/mini/model000600161.pt --text_prompt "greet a friend" --babel_prompt "hug"
```
<!-- ```shell
python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --text_prompt "the person walked forward and is picking up his toolbox."
``` -->


**You may also define:**
* `--device` id.
* `--seed` to sample different prompts.
* `--motion_length` (text-to-motion only) in seconds (maximum is 9.8[sec]).
* `--second_llm`

**Running those will get you:**

* `results.npy` file with text prompts and xyz positions of the generated animation
* `sample##_rep##.mp4` - a stick figure animation for each generated motion.

It will look something like this:

![example](assets/example_stick_fig.gif)

You can stop here, or render the SMPL mesh using the following script.

### Render SMPL mesh

To create SMPL mesh per frame run:

```shell
python -m visualize.render_mesh --input_path /path/to/mp4/stick/figure/file
```

**This script outputs:**
* `sample##_rep##_smpl_params.npy` - SMPL parameters (thetas, root translations, vertices and faces)
* `sample##_rep##_obj` - Mesh per frame in `.obj` format.



## Acknowledgments

This code is heavily adapted from:

- [MDM](https://github.com/GuyTevet/motion-diffusion-model)

- [HumanML3D](https://github.com/EricGuoICT/HumanML3D)

- [MotionCLIP](https://github.com/GuyTevet/MotionCLIP)

- [SMPL(-X)](https://github.com/vchoutas/smplx)

