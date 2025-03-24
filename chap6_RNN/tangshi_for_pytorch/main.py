import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import rnn as rnn_lstm

# 使用 GPU 训练
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 特殊符号
start_token = 'S'
end_token = 'E'
pad_token = ' '
unkown_token = 'U'
split_token = '，'
final_token = '。'
disabled_tokens = ['_', '(', '（', '《', '[', start_token, end_token]


# 设置超参数
MIN_WORD_FREQ = 5
BATCH_SIZE = 64
embedding_dim = 164
lstm_hidden_dim = 128

def process_poems(file_name, start_token=start_token, end_token=end_token):
    poems = []
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                content = line.strip().split(':')[1] if ':' in line else line.strip()
                # content = content.replace(' ', '').replace('，', '').replace('。', '')
                content = content.replace(' ', '')
                if any(x in content for x in disabled_tokens) or len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token # start_token+ ?
                poems.append(content)
            except Exception as e:
                print(f"Error processing line: {line}, error: {e}")
    
    poems = sorted(poems, key=lambda p: len(p))
    all_words = [word for poem in poems for word in poem]
    counter = collections.Counter(all_words)
    # 除去频率低于MIN_WORD_FREQ的词
    counter = {k: v for k, v in counter.items() if v >= MIN_WORD_FREQ}
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words + (pad_token, unkown_token)  # 添加特殊符号到词汇表中
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(lambda word: to_int(word, word_int_map), poem)) for poem in poems]

    return poems_vector, word_int_map, words

def generate_batch_generator(poems_vec, batch_size, pad_token_id):
    n_chunk = len(poems_vec) // batch_size
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        
        # 找出当前batch中最长的诗的长度
        max_len = max(len(p) for p in x_data)
        
        # 对每首诗进行填充，使其长度一致
        x_data_padded = [p[:-1] + [pad_token_id]*(max_len - len(p[:-1])) for p in x_data]  # 使用填充
        y_data_padded = [p[1:] + [pad_token_id]*(max_len - len(p[1:])) for p in x_data]  # 同样对y也进行填充

        yield np.array(x_data_padded, dtype=np.int64), np.array(y_data_padded, dtype=np.int64)


def run_training():
    # 加载并处理数据集
    poems_vector, word_to_int, vocabularies = process_poems('./poems.txt')
    
    # 初始化模型、损失函数和优化器
    word_embedding = rnn_lstm.word_embedding(vocab_length=len(word_to_int) + 1, embedding_dim=embedding_dim)
    rnn_model = rnn_lstm.RNN_model(batch_sz=BATCH_SIZE, vocab_len=len(word_to_int) + 1, word_embedding=word_embedding,
                                   embedding_dim=embedding_dim, lstm_hidden_dim=lstm_hidden_dim, device=device)
    rnn_model.to(device)
    
    optimizer = optim.Adam(rnn_model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    loss_fun = torch.nn.CrossEntropyLoss(ignore_index=word_to_int[pad_token])  # 使用 CrossEntropyLoss 并忽略填充部分
    # rnn_model.load_state_dict(torch.load('./poem_generator_rnn')) # 加载已训练好的模型

    batch = 0  # 初始化批次计数器

    print(word_to_int['S'])
    print(word_to_int['E'])
    print(word_to_int['，'])
    print(word_to_int['。'])
    print(word_to_int[unkown_token])
    print(word_to_int[pad_token])

    # 训练模型
    rnn_model.train()  # 设置为训练模式
    for epoch in range(30):
        for batch_x, batch_y in generate_batch_generator(poems_vector, BATCH_SIZE, word_to_int[pad_token]):
            x = Variable(torch.from_numpy(batch_x)).to(device)
            y = Variable(torch.from_numpy(batch_y)).to(device)
                
            pre, _ = rnn_model(x)

            print(x.shape, pre.shape, y.shape)

            loss = loss_fun(pre.permute(0, 2, 1), y) 

            _, pre = torch.max(pre, dim=-1)
            print('Input:', x.data.tolist()[0])
            print('Prediction:', pre.data.tolist()[0])
            print('Actual:', y.data.tolist()[0])
            print('*' * 30)
            
            print("Epoch:", epoch, "Batch:", batch, "Loss:", loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 1) # rnn存在梯度爆炸或消失问题，使用梯度裁剪解决
            optimizer.step()
            
            if batch % 20 == 0:
                torch.save(rnn_model.state_dict(), './poem_generator_rnn')
                print("Model saved.")
            
            batch += 1  # 在每个批次处理后递增批次计数器
        
        scheduler.step() # 更新学习率


def to_word(predict, vocabs):  # 预测的结果转化成汉字
    sample = np.argmax(predict)

    if sample >= len(vocabs):
        sample = len(vocabs) - 1

    return vocabs[sample]

def random_sample_to_word(predict, vocabs, sample_num=50):  
    # 随机在概率最高的50个字中采样一个，并预测的结果转化成汉字
    samples = np.argsort(predict)[-sample_num:]
    samples = [s for s in samples if s < len(vocabs)]

    if any(token in map(vocabs.__getitem__,samples) for token in [start_token, unkown_token, pad_token, split_token, final_token]):
        samples = [s for s in samples if vocabs[s] not in [start_token, unkown_token, pad_token, split_token, final_token]]
    sample = np.random.choice(samples)

    return vocabs[sample]


def to_int(word, vocabs: dict):  # 汉字转化成数字
    return vocabs.get(word, vocabs.get(unkown_token))


def pretty_print_poem(poem):  # 令打印的结果更工整
    print(poem.replace(start_token, '').replace(end_token, ''))
    

def gen_poem(begin_word, each_sentence_len=8, total_sentence_num=4):
    _, word_int_map, vocabularies = process_poems('./poems.txt')
    word_embedding = rnn_lstm.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=embedding_dim)
    rnn_model = rnn_lstm.RNN_model(batch_sz=BATCH_SIZE, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,
                                   embedding_dim=embedding_dim, lstm_hidden_dim=lstm_hidden_dim, device=device)

    rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))

    rnn_model.to(device)
    rnn_model.eval()  # 设置为测试模式

    # 指定开始的字
    poem = begin_word
    word = begin_word

    unkown_flag = False

    if begin_word not in word_int_map:
        word = unkown_token
        unkown_flag = True

    while word != end_token:
        input = np.array([word_int_map[w] for w in poem],dtype= np.int64)
        input = Variable(torch.from_numpy(input))
        output,_ = rnn_model(input ,is_test=True)
        word = random_sample_to_word(output.data.tolist()[-1], vocabularies)
        poem += word
        if (len(poem)+1) % each_sentence_len ==0:
            poem += '，'
        if len(poem) >= each_sentence_len * total_sentence_num:
            poem = poem[:-1]+ '。'
            break

    if unkown_flag:
        poem = begin_word + poem[1:]
    return poem



# run_training()  # 如果不是训练阶段 ，请注销这一行 。 网络训练时间很长。


pretty_print_poem(gen_poem("日"))
pretty_print_poem(gen_poem("红"))
pretty_print_poem(gen_poem("山"))
pretty_print_poem(gen_poem("夜"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("君"))