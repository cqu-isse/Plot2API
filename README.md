# Plot2API
Plot2API: Recommending Graphic API from Plot via Semantic Parsing Guided Neural Network

Accepted by SANER2021

# Dataset
![The data distribution of the Python-Plot13 dataset:](https://github.com/cqu-isse/Plot2API/blob/master/data/python.png)
![The data distribution of the R-Plot32 dataset:](https://github.com/cqu-isse/Plot2API/blob/master/data/R.png)

# Note
Firstly, you should run the EfficientNet successfully. The link is https://github.com/lukemelas/EfficientNet-PyTorch. 
Then download the datasets (https://pan.baidu.com/s/1I8btvuLwn5w3GnI-ZCV3Ew, the code is ISSE).
The result will be a little different from the paper due to initialization.

## Run
- python>=3.6

$ pip install -r requirements.txt

$ python SPGNN.py -g 0
