import os
import random

random.seed(123)


def get_emo(emo_count):
    total = sum(emo_count)
    mark = random.randint(1, total)
    count = 0
    for i in range(len(emo_count)):
        count += emo_count[i]
        if count >= mark:
            return i


def get_uniq_src(data_path, type):
    f = open(os.path.join(data_path, type + '.txt'), 'r', encoding='utf8')
    o = open(os.path.join(data_path, type + '_uniq_src.txt'), 'w', encoding='utf8')

    
    
    emo_tgt = [0] * 6
    emo_count = [0] * 6
    count = [0] * 6

    lines = f.readlines()

    if len(lines[-1]) == 0:
        lines = lines[:-1]

    for i in range(len(lines)):
        cur_line = lines[i]
        src, src_emo, tgt, tgt_emo = cur_line.strip('\n').split('\t')
        emo_count[int(tgt_emo)] = emo_count[int(tgt_emo)] + 1
        emo_tgt[int(tgt_emo)] = tgt

        
        if i + 1 != len(lines):
            next_line = lines[i + 1]
            next_src, _, _, _ = next_line.strip('\n').split('\t')
            if src == next_src:
                continue
            else:
                tgt_emo = get_emo(emo_count)
                tgt = emo_tgt[tgt_emo]
                o.write(src + '\t' + str(src_emo) + '\t' + tgt + '\t' + str(tgt_emo) + '\n')
                count[tgt_emo] += 1
                print(f'{src}:\n{emo_count}, {tgt_emo}')
                emo_count = [0] * 6
        else:
            tgt_emo = get_emo(emo_count)
            tgt = emo_tgt[tgt_emo]
            o.write(src + '\t' + str(src_emo) + '\t' + tgt + '\t' + str(tgt_emo) + '\n')
            count[tgt_emo] += 1
            print(f'{src}:\n{emo_count}, {tgt_emo}')
    print('total:', count)


if __name__ == '__main__':
    
    get_uniq_src('../../ECM/data/ESTC', 'test')
    get_uniq_src('../../ECM/data/ESTC', 'valid')
