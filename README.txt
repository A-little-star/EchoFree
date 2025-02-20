说明

- 该目录共包含2024年底交付的4个模型，分别是rnnvqe_last, rnnvqe_last_stream(流式版本), rnnvqe_last_qat(量化版本), rnnvqe_last_qat_stream(量化模型的流式版本)
- 包含了训练脚本和测试脚本，训练脚本包括非量化和量化模型两个脚本，在./scripts/train.sh中；测试脚本有4个，分别对应上述的4个模型，在./scripts/test.sh中有说明
- 包含AECMOS的计算代码，位于./AECMOS_local目录下
- 包含了现有模型rnnvqe_last的C语言版本，在rnnvqe_c_based目录下，对应的说明位于rnnvqe_c_based目录下的README中

测试时，只需要将数据准备为goertek.lst文件中的格式（混合音频以"_mic"为后缀，参考音频以"_lpb"为后缀，
两种音频位于同一目录下，.lst文件中只记录混合音频路径），
然后在./scirpts/test.sh中选用对应的模型，执行./scripts/test.sh脚本即可。
如果需要修改.lst文件的路径，修改对应的.yml配置文件中的test_list:goer_list一项表示的路径即可。
