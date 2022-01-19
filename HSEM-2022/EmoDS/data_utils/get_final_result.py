
for i in ['like','sadness','happiness','disgust','anger']:
    f1 = open(f'../human_evaluation/test.response_{i}.txt.epoch_15.', 'r', encoding='utf-8')
    f2 = open(f'../human_evaluation/{i}.txt', 'w', encoding='utf-8')
# for i in range(0,30):
    # f1 = open(f'../result4/test.response.txt.epoch_{i}.', 'r', encoding='utf-8')
    # f2 = open(f'../result4/result_epoch_{i}.txt', 'w', encoding='utf-8')
    start_symbol = "<ss>"
    end_symbol = "<es>"

    for line in f1.readlines():
        words, emotion = line.strip().split('\t')
        words = words.strip().split()
        if start_symbol in words:
            start_index = words.index(start_symbol) + 1
        else:
            start_index = 0
        if end_symbol in words:
            end_index = words.index(end_symbol)
        else:
            end_index = len(words)

        selected_words = words[start_index: end_index]
        sentence = " ".join(selected_words)
        f2.write(sentence)
        f2.write('\t'+emotion)
        f2.write("\n")
