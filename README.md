
## 🌳 Code Structure

```
├─MVLPNet
|   ├─utils.py
|   ├─vis.py # vis.py is the code for visualization
|   ├─test.py
|   ├─test.sh
|   ├─train.py
|   ├─train.sh
|   ├─train_base.py
|   ├─train_base.sh
|   ├─util
|   ├─model
|   |   ├─workdir
|   |   ├─util
|   |   ├─few_seg
|   |   |    └MVLPNet.py
|   |   ├─backbone
|   |   ├─clip
|   ├─lists
|   ├─initmodel
|   |     ├─PSPNet
|   |     ├─CLIP
|   ├─vgg16_bn.pth
|   ├─resnet50_v2.pth
|   ├─exp
|   ├─dataset
|   ├─config
├─data
|  ├─iSAID
|  |   ├─train.txt
|  |   ├─val.txt
|  |   ├─img_dir
|  |   ├─ann_dir
```

## 📝 Data Preparation

- Create a folder `data` at the same level as this repo in the root directory.

  ```
  cd ..
  mkdir data
  ```
- iSAID_512:
iSAID.tar.gz : https://pan.baidu.com/s/11ZhZ01KVjfPyHcoZ2MkfeA password: 0000

- iSAID_256:
iSAID.tar.gz : https://pan.baidu.com/s/1WgZBH075gjmypS4NbiLaXg password: 0000 

- LoveDA:
LoveDA.tar.gz : https://pan.baidu.com/s/1XG7zsh5uTOerffrE73cj2g password: 0000 

## Train

### Training base-learners (two options)

- Option 1: training from scratch

  Download the pre-trained backbones from (https://pan.baidu.com/s/1tWAUKYvP-sh_LcCOy1-P7Q password: 0000) and put them into the `MVLPNet/initmodel` directory.
  The clip model is placed in the `MVLPNet/initmodel` directory: (https://pan.baidu.com/s/1vwtIinePOP7UdhrEDj4HKg password: 0000)
  ```
  sh train_base.sh
  ```
- Option 2: loading the trained models
  
  ```
  mkdir initmodel
  cd initmodel
  ```
  
  Put the provided (https://pan.baidu.com/s/1I4s8PLy4N5Qb7UeE7VsVXQ password: 0000) in the newly created folder `initmodel` and rename the downloaded file to `PSPNet`, *i.e.*, `MVLPNet/initmodel/PSPNet`.

### Training few-shot models

To train a model, run

```
sh train.sh
```

### Testing few-shot models

To evaluate the trained models, run

```
sh test.sh
```


## 👏 Acknowledgements
The project is based on [PFENet](https://github.com/dvlab-research/PFENet) , [R2Net](https://github.com/chunbolang/R2Net) and [PI-CLIP](https://github.com/vangjin/PI-CLIP). Thanks for the authors for their efforts.

