```
/project_root
|-- data/
|   |-- starcraft_data.json
|-- config.py             # 存放所有配置和超参数
|-- prepare_data.py       # 数据预处理和分词器准备
|-- model.py              # 自定义模型结构 (带MLP的Qwen)
|-- data_collator.py      # 自定义数据整理器
|-- train.py              # 核心训练脚本
|-- inference.py          # 推理测试脚本
|-- run_training.sh       # (可选) 用于启动多卡训练的shell脚本
|-- requirements.txt      # 项目依赖
```