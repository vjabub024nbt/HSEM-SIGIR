### steps:
##### * train classfifier
    * extract_data
    * data_preprocess
    * build_vocab
    * train
##### * annotate dataset
    *annotate_stc -> ESTC
##### * get test input（Post + Emotion label）：
    * get_uniq_src_sentence -> test_uniq_src/valid_uniq_src
##### * input uniq_src and predict response
    * chatbot generate responses -> response.txt
##### * if no GT label, add it（label from uniq_src）
    * combine_response_true_label
##### * compute emotion accuracy
    * test -> emotion acc