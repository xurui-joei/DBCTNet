# DBCTNet
## DBCTNet: Double Branch Convolution-Transformer Network for Hyperspectral Image Classification
"DBCTNet: Double Branch Convolution-Transformer Network for Hyperspectral Image Classification" has been accepted by TGRS in Feb 2024, which is a pytorch implementation of DBCTNet.

## Requirements
To install the required dependencies, please run the following command:
``` bash
cd DBCTNet
pip install -r requirements.txt
```
## Train and test
To run the program, please execute the script below:
```bash
sh run.sh
```
## Others
You can add your own HSI dataset to data/ and modify the train_test.py to train DBCTNet on your own dataset. The weight of model will be saved at weights/. 

## Citation
If you find this project useful for your research or if you use it in your academic projects, we would appreciate it if you could cite it as follows:

```bibtex
@ARTICLE{DBCTNet,
  author={Xu, Rui and Dong, Xue-Mei and Li, Weijie and Peng, Jiangtao and Sun, Weiwei and Xu, Yi},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={DBCTNet: Double Branch Convolution-Transformer Network for Hyperspectral Image Classification}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
  keywords={Feature extraction;Transformers;Convolution;Three-dimensional displays;Convolutional neural networks;Kernel;Standards;Convolutional neural networks (CNNs);hyperspectral image (HIS) classification;multiscale;Transformer},
  doi={10.1109/TGRS.2024.3368141}}


