# chef-ai
A series of AI models with the goal of generating recipes using the [food.com dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).

This is split into two major steps, which will be trained individually.
1. Generate a list of ingredients
2. Depending on the ingredients, generate a list of instructions

A transformer trained on 140000 recipes can be downloaded [here](https://drive.google.com/file/d/18zf-OO0fC4bsqWaDUpoXrjLSvu4rBp1R/view?usp=sharing).
- Move and rename the .pkl file to Models/Instructions/Transformer.pkl
- Ensure your TransformerTrainer is set to these parameters TransformerTrainer(200, 3, 500, 400, 4, 0.0)
- Test it out using `python main.py test`

An attempt was certainly made. But **for your own health, don't follow its instructions**:
![Attempt](https://user-images.githubusercontent.com/19623152/229424635-f588ecbb-9b91-4b16-a37e-bfd120c3cfd1.gif)


## References
- MLP, [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- CNN, [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499)
- RNN, [Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- LSTM, [Graves et al. 2014](https://arxiv.org/abs/1308.0850)
- GRU, [Kyunghyun Cho et al. 2014](https://arxiv.org/abs/1409.1259)
- Transformer, [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)
- Makemore, [Andrej Karpathy 2022](https://github.com/karpathy/makemore)
- Permutation-equivariant neural networks, [Guttenberg et al. 2016](https://arxiv.org/pdf/1612.04530.pdf)
