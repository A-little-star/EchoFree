# RNNVQE 基于C语言实现帧级别流式推理

这里，将pytorch训练的声学回声消除模型转为C语言实现。

在./include/tensor.h中，对Tensor进行了定义，并给出了pytorch中对Tensor的常见操作的声明，它们的具体实现在./libs/tensor.c中。
在./include/module.h中，对pytorch中常用的模块（如Conv2d, GRU, Linear等）进行了声明，它们的具体实现在./module/下的子文件中。
在./model/rnnvqe.c中，定义了完整的深度学习模型RNNVQE，它接受mic（混合信号）和y（线性滤波的回声估计）的Bark特征作为输入，输出预测的近端语音的Bark谱增益。

在./model/main.c中，给出了非流式推理和流式推理两种推理代码。同时给出了根据.lst文件逐条推理的代码（非流式推理和流式推理单条样例代码在注释中）。

推理时，只需要将数据准备为goertek.lst文件中的格式（混合音频以"_mic"为后缀，参考音频以"_lpb"为后缀，两种音频位于同一目录下，.lst文件中只记录混合音频路径），然后在项目根目录./中执行make编译源文件，然后执行make run执行程序。
重新编译时，以以下顺序执行：
make clean
make
make run