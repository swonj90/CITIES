# CITIES: Contextual Inference of Tail-Item Embeddings for Sequential Recommendation

PyTorch implementation of the paper: "CITIES: Contextual Inference of Tail-Item Embeddings for Sequential Recommendation", ICDM, 2020.

## Abstract
Sequential recommendation techniques provide users with product recommendations fitting their current prefer- ences by handling dynamic user preferences over time. Previous studies have focused on modeling sequential dynamics without much regard to which of the best-selling products (i.e., head items) or niche products (i.e., tail items) should be recommended. We scrutinize the structural reason for why tail items are barely served in the current sequential recommendation model, which consists of an item-embedding layer, a sequence-modeling layer, and a recommendation layer. Well-designed sequence-modeling and recommendation layers are expected to naturally learn suit- able item embeddings. However, tail items are likely to fall short of this expectation because the current model structure is not suitable for learning high-quality embeddings with insufficient data. Thus, tail items are rarely recommended. To eliminate this issue, we propose a framework called CITIES, which aims to enhance the quality of the tail-item embeddings by training an embedding-inference function using multiple contextual head items so that the recommendation performance improves for not only the tail items but also for the head items. Moreover, our framework can infer new-item embeddings without an additional learning process. Extensive experiments on two real- world datasets show that applying CITIES to the state-of-the-art methods improves recommendation performance for both tail and head items. We conduct an additional experiment to verify that CITIES can infer suitable new-item embeddings as well.

## Usage
### Preparing dataset and pre-trained item embedding
1. Download 'data' and 'pretrained_model' from the following link: https://drive.google.com/drive/folders/1wjsj70kYfOd166zbD1s4OX6JmZ2ipa3j?usp=sharing

2. Unzip the downloaded folders to the cloned code directory.  
cities.py  
...  
utils.py  
data/yelp.csv  
pretrained_model/yelp/bert.pth  


### How to run the code
```python main.py --dataset yelp --pretrained_model bert --embedding_key bert.embedding.token.weight```
1. If you'd like to use a different dataset, save your dataset to the 'data' directory and match the format to yelp.csv.  
   Then, change the --dataset argument to the name of your dataset.  
2. If you'd like to use a different pretrained item embedding, pretrain your model and save the model to the 'pretrained_model' directory. (Note that you have to write the model with the '.pth' extension.)   
   Then, change the --pretrained_model argument to the name of your model and --embedding_key argument to the name of item embedding's name. 
   
   
## Citation
If you use this code, please cite the paper.
```
@inproceedings{jang2020cities,
  title={CITIES: Contextual Inference of Tail-item Embeddings for Sequential Recommendation},
  author={Jang, Seongwon and Lee, Hoyeop and Cho, Hyunsouk and Chung, Sehee},
  booktitle={2020 IEEE International Conference on Data Mining (ICDM)},
  pages={202--211},
  year={2020},
  organization={IEEE}
}
```
