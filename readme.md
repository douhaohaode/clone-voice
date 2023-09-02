# 🐶 tts和语音克隆gui工具

## 简介
 基于[Bark](https://github.com/suno-ai/bark)和[Gradio Web UI](https://github.com/gradio-app/gradio) 的文字转换语音和语音克隆工具

## ⚠ 免责声明

此工具是为了研究和学习目的开发的,它可能会以意想不到的方式偏离所提供的提示,对生成的任何输出不承担任何责任, 使用风险自负,请负责任地行事。

## tts
  基于[Bark](https://github.com/suno-ai/bark)的文字转语音功能, 支持特殊文字输入 ,支持多国语言提示库 [Bark语言提示库](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c) 

## 克隆
  基于[bark-voice-cloning-HuBERT-quantizer](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer)开发的工具进行克隆, 目前支持克隆语言 英语、日语、德语、国语训练中希望不会难产
  
 #### 下载对应模型并且重命名放到对应文件夹下如图:

  ![models.png](asset%2Fmodels.png)    

## 下载相关模型放入到对应的models文件下
### 官方

| 名字                                                                                                                                                        | HuBERT Model                                                          | Quantizer Version | Epoch | Language | Dataset                                                                                          |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|-------------------|-------|----------|--------------------------------------------------------------------------------------------------|
| [quantifier_V1_hubert_base_ls960_23.pth/en_tokenizer](https://huggingface.co/GitMylo/bark-voice-cloning/blob/main/quantifier_V1_hubert_base_ls960_23.pth) | [hubbert](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) | 1                 | 23    | ENG      | [GitMylo/bark-semantic-training](https://huggingface.co/datasets/GitMylo/bark-semantic-training) |

### 社区

| 作者                                          | 姓名                                                                                                                                                                                    | HuBERT Model                                                              | Quantizer Version | Epoch | Language | Dataset                                                                                                                      |
|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|-------------------|-------|----------|------------------------------------------------------------------------------------------------------------------------------|
| [junwchina](https://github.com/junwchina)       | [japanese-HuBERT-quantizer_24_epoch.pth/ja_tokenizer](https://huggingface.co/junwchina/bark-voice-cloning-japanese-HuBERT-quantizer/blob/main/japanese-HuBERT-quantizer_24_epoch.pth) | [HuBERT Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) | 1                 | 8     | JA       | [Hobis/bark-polish-semantic-wav-training](https://huggingface.co/datasets/junwchina/bark-japanese-semantic-wav-training/tree/main)           |
| [C0untFloyd](https://github.com/C0untFloyd) | [ german-HuBERT-quantizer_14_epoch.pth/pl_tokenizer](https://huggingface.co/CountFloyd/bark-voice-cloning-german-HuBERT-quantizer/blob/main/german-HuBERT-quantizer_14_epoch.pth)     | [HuBERT Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) | 1                 | 14    | GER      | [CountFloyd/bark-german-semantic-wav-training](https://huggingface.co/datasets/CountFloyd/bark-german-semantic-wav-training) |



## 训练 
  克隆不同国家的语言需要不同国家对应的tokenizer.pth文件，比如要训练中文那么就需要一个对应中文的tokenizer.pth文件，需要对一门语言进行训练。

###  训练步骤:
     1.数据准备如图:
![trian1.png](asset%2Ftrian1.png)

     2.数据处理
![train2.png](asset%2Ftrain2.png)

     3.开始训练

## 使用

  1.python环境  python>3.10 
  
  2.依次执行下列命令：
  ```python
  git clone https://github.com/douhaohaode/clone-voice.git
  cd clone_vicoe
  pip install . 
  pip install -r requirements.txt
  python webui
```

## 感谢
  此工程参考和使用了一些开源库和一些模型感谢他们的作者以及开源精神

 * [suno-ai/bark](https://github.com/suno-ai/bark)
 * [gitmylo/bark-voice-cloning-HuBERT-quantizerr](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer)
 * [C0untFloyd/bark-gui](https://github.com/C0untFloyd/bark-gui)