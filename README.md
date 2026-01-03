# 《人工神经网络》大作业报告——论文复现

**Versatile-Diffusion rebuild in jittor**

```
王一鸣 
2022011305 wangyimi22@mails.tsinghua.edu.cn

谢宇涵
2022011309 xie-yh22@mails.tsinghua.edu.cn

陈道凡
2022011312 chendf22@mails.tsinghua.edu.cn
```

## 项目介绍

本项目复现了论文《Versatile Diffusion: Text, Image and Video Generation with a Single Diffusion Model》的部分内容，使用jittor框架重写了论文中核心代码，并在部分数据集上进行了训练和测试。

## 开发环境

```
ubuntu 20.04
GPU: L20 * 1（48GB）
```

## 环境配置和测试

在项目根目录下运行以下命令安装依赖和测试：
```
conda create -f env.yaml
conda activate Versatile-Diffusion
python app.py
```

## 项目结构

```py
.
|-- README.md
|-- app.py
|-- assets # contains demo images and figures
|   |-- demo
|   |   |-- mcg_example
|   |   |   |-- e0i0.jpg
|   |   |   |-- e0i1.jpg
|   |   |   `-- e0i2.jpg
|   |   |-- misc
|   |   |   |-- mask_inst1.gif
|   |   |   |-- mask_inst2.gif
|   |   |   |-- mask_inst3.gif
|   |   |   `-- noimage.jpg
|   |   |-- reg_example
|   |   |   |-- benz.jpg
|   |   |   |-- boy_and_girl.jpg
|   |   |   |-- church.jpg
|   |   |   |-- firework.jpg
|   |   |   |-- ghibli.jpg
|   |   |   |-- horse.jpg
|   |   |   |-- house_by_lake.jpg
|   |   |   |-- matisse.jpg
|   |   |   |-- night_light.jpg
|   |   |   |-- noimage.jpg
|   |   |   |-- paris.jpg
|   |   |   |-- penguin.jpg
|   |   |   |-- san_diego.jpg
|   |   |   |-- scream.jpg
|   |   |   |-- space.jpg
|   |   |   |-- tiger.jpg
|   |   |   |-- train.jpg
|   |   |   `-- vermeer.jpg
|   |   |-- tcg_example
|   |   |   |-- e0i0.jpg
|   |   |   |-- e0i1.jpg
|   |   |   |-- e1i0.jpg
|   |   |   |-- e1i1.jpg
|   |   |   |-- e2i0.jpg
|   |   |   |-- ghibli_mask.png
|   |   |   `-- space_mask.png
|   |   `-- temp
|   |       `-- dummy_file_to_git_sync_the_folder.txt
|   `-- figures
|       |-- gallary_dual_guided.png
|       |-- gallary_i2i_1.png
|       |-- gallary_i2i_2.png
|       |-- gallary_t2i.png
|       |-- qcompare1.png
|       |-- qcompare2.png
|       |-- qcompare3.png
|       |-- teaser.png
|       `-- vd_combined.png
|-- configs # configuration files
|   `-- model
|       |-- autokl.yaml
|       |-- clip.yaml
|       |-- image_to_text_finetune.yaml
|       |-- image_var_finetune.yaml
|       |-- openai_unet.yaml
|       |-- optimus.yaml
|       |-- text_to_image_finetune.yaml
|       |-- text_var_finetune.yaml
|       `-- vd.yaml
|-- cusomized_gradio_blocks.py
|-- env.yaml
|-- lib # core library, store model structures and training/evaluation stages
|   |-- __init__.py
|   |-- cfg_helper.py
|   |-- cfg_holder.py
|   |-- data_factory
|   |   |-- __init__.py
|   |   |-- my_dataset_tar.py
|   |   `-- my_vd_dataset.py
|   |-- evaluator
|   |   `-- evaluator.py
|   |-- image_to_text_eval_stage.py
|   |-- image_to_text_train_stage.py
|   |-- image_var_eval_stage.py
|   |-- image_var_train_stage.py
|   |-- log_service.py
|   |-- model_zoo
|   |   |-- __init__.py
|   |   |-- common
|   |   |   |-- get_model.py
|   |   |   |-- get_optimizer.py
|   |   |   |-- get_scheduler.py
|   |   |   `-- utils.py
|   |   |-- model_jittor
|   |   |   |-- attention.py
|   |   |   |-- ddim.py
|   |   |   |-- diffusion_utils.py
|   |   |   |-- ema.py
|   |   |   |-- nn_compat.py
|   |   |   |-- openaimodel.py
|   |   |   `-- vd.py
|   |   `-- model_torch
|   |       |-- autokl.py
|   |       |-- autokl_modules.py
|   |       |-- autokl_utils.py
|   |       |-- clip.py
|   |       |-- optimus.py
|   |       |-- optimus_models
|   |       |   |-- configuration_bert.py
|   |       |   |-- configuration_gpt2.py
|   |       |   |-- configuration_utils.py
|   |       |   |-- file_utils.py
|   |       |   |-- modeling_utils.py
|   |       |   |-- optimus_bert.py
|   |       |   |-- optimus_gpt2.py
|   |       |   |-- tokenization_bert.py
|   |       |   |-- tokenization_gpt2.py
|   |       |   |-- tokenization_utils.py
|   |       |   `-- vocab
|   |       |       |-- bert-base-cased-vocab.txt
|   |       |       |-- bert_vocab_download_info.json
|   |       |       |-- gpt2-merges.txt
|   |       |       |-- gpt2-vocab.json
|   |       |       `-- gpt2_vocab_merge_download_info.json
|   |       `-- torch_hub.py
|   |-- sync.py
|   |-- text_to_image_eval_stage.py
|   |-- text_to_image_train_stage.py
|   |-- text_var_eval_stage.py
|   |-- text_var_train_stage.py
|   `-- utils.py
|-- main.py # training and evaluation entry
```

其中，model_torch文件夹下的代码均为从原作者的PyTorch实现中直接移植而来，均为使用的gpt2和bert模型等encoder所需，并不属于versatile-diffusion原创模型的一部分，且并未在训练时进行更新。因此，我们仅对versatile-diffusion中的核心代码进行了复现和测试，包括 diffuser image, diffuser text, diffuser global等模块，以及相应的训练和评估流程。

## Pretrained models

所有的预训练模型均存放在pretrained文件夹下，可从[此处](https://huggingface.co/shi-labs/versatile-diffusion/tree/main/pretrained_pth)下载，结构如下：

```py
├── pretrained
│   └── kl-f8.pth
│   └── optimus-vae.pth
│   └── vd-four-flow-v1-0.pth
│   └── vd-four-flow-v1-0-fp16.pth
```
其中前两个为自动编码器和文本编码器的预训练模型，后两个为versatile-diffusion模型的预训练模型。

## 数据结构

数据的组织结构需要根据具体的train stage和eval stage进行调整，对于项目中我们已经实现的train stage和eval stage，数据结构如下所示：

```py
├── data
│    └── train
│           └── images # contains training images
│           └── captions.jsonl
│    └── eval
│           └── images # contains evaluation images
│           └── captions.jsonl
```

## 训练代码

在项目根目录下运行以下命令进行text to image的微调训练：
```
python main.py --config configs/model/text_to_image_finetune.yaml
```
其中，configs/model/text_to_image_finetune.yaml为训练配置文件，可根据需要进行修改，需要实现具体的train stage和eval stage，具体需要的参数可在配置文件中查看。
由于整体的模型较大，经过实际测试，无法同时训练两条以上的训练流水线，因此在配置文件中，我们将train stages和eval stages均设置为单条流水线，也即在训练完一条流水线后保存训练好的参数，再加载上一条流水线训练后的参数，进行下一条流水线的训练。如果需要同时训练多条流水线，可以将配置文件中的train stages和eval stages进行相应的修改。