
# Bio-inspired colour opponency convolutional neural networks 

***


1. Clone this [Colour Opponency BioNet](https://github.com/BlinkingStalker/ColourOpponencyGaborNet)
2. Clone the [CIFAR-10G]((https://github.com/bdevans/CIFAR-10G)) for generalisation testset.
3. Clone the [Ecoset](https://codeocean.com/capsule/9570390/tree/v1) If you encounter problems for download, you may try a direct download from CodeOcean's S3 bucket:
"aws s3 cp --no-sign-request s3://codeocean-datasets/0ab003f4-ff2d-4de3-b4f8-b6e349c0e5e5/ecoset.zip ."
4. Extract categories as CIFAR-10, create Ecoset-10.[(could use this script)](https://github.com/bdevans/BioNet/blob/main/scripts/make_ecoset-cifar10.sh)

Expected directory structure
----------------------------
```
.
├── bionet
│   ├── Introduction.ipynb
│   ├── OpponencyCifarVGG16.ipynb
│   ├── OpponencyCifarResnet.ipynb
│   ├── OpponencyEcosetVGG16.ipynb
│   ├── OpponencyEcosetResnet.ipynb
│   ├── ResNet.py
│   └── VGG.py

├── data
│   ├── CIFAR-10G
│   ├── ecoset
│   └── ecoset-cifar10
├── logs
├── models
├── results
├── scripts
└── README.md
```

Introduction about Gabor filter with different parameters.
------------------------------
Click here
[Introduction of the gabor filter](https://github.com/BlinkingStalker/ColourOpponencyGaborNet/bionet/Introduction.ipynb)


Training and testing the model
------------------------------
The main script to handle training and testing are 4 .ipynb files. in the ./bionet dictionary. 
The general Colour Opponency GaborNet for VGG and Resnet are in file VGG.py and Resnet.py. 
The saved model could be found in dictionary `models`. 


