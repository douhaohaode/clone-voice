# 🐶 tts和语音克隆gui工具

## 简介
 基于[Bark](https://github.com/suno-ai/bark)和[Gradio Web UI](https://github.com/gradio-app/gradio) 的文字转换语音和语音克隆工具

## ⚠ 免责声明

此工具是为了研究和学习目的开发的,它可能会以意想不到的方式偏离所提供的提示,对生成的任何输出不承担任何责任, 使用风险自负,请负责任地行事。

## tts
  基于[Bark](https://github.com/suno-ai/bark)的文字转语音功能, 支持特殊文字输入 ,支持多国语言提示库 [Bark语言提示库](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c) 

## 克隆和训练
  基于[Bark](https://github.com/suno-ai/bark)和[bark-voice-cloning-HuBERT-quantizer](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer)开发的工具进行克隆, 目前支持克隆语言 英语、日语、德语、国语训练中希望不会难产

## 使用
  ```python
  git clone https://github.com/douhaohaode/clone-vicoe-gui
  cd clone_vicoe-gui
  pip install . 
  pip install -r requirements.txt
```

## 感谢
  此工程参考和使用了一些开源库和一些模型感谢他们的作者以及开源精神

 * [suno-ai/bark](https://github.com/suno-ai/bark)
 * [gitmylo/bark-voice-cloning-HuBERT-quantizerr](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer)
 * [C0untFloyd/bark-gui](https://github.com/C0untFloyd/bark-gui)