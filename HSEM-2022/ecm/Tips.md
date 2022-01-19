* Train

	``` python baseline.py --use_emb --use_imemory --use_ememory ```

You can remove "--use_emb", "--use_imemory", "--use_ememory" to remove the embedding, internal memory, and external memory module respectively. The model will achieve the expected performance after 20 epochs.

* Test

	``` PYTHONIOENCODING=utf8 python baseline.py --use_emb --use_imemory --use_ememory --decode	```

You can test and apply the ecm model using this command. Note: the input words should be splitted by ' ', for example, '我 很 喜欢 你 ！', or you can add the chinese text segmentation module in split() function.
