from scipy.stats import pearsonr


def read_vocab(file_name):
  f = open(file_name, "r")
  my_list = {}
  vocab_list = []
  scores = []
  f.readline()
  for line in f:
      words = line.split(",")

      if words[1] not in my_list:
          scores.append(float(words[2]))

      my_list[words[1].lower()]= float(words[2])
      vocab_list.append(words[1].lower())


  f.close()
  return my_list, vocab_list, scores


def read_vocab2(file_name, vocab_list, x1_dict):
  f = open(file_name, "r")
  my_list = {}
  ordered_list = []
  ordered_list_2 = []
  f.readline()
  for line in f:
      words = line.split(",")
      if words[1].lower() in vocab_list:
          my_list[words[1].lower()] = float(words[2])

  for word in vocab_list:
      ordered_list.append(my_list[word])
      ordered_list_2.append(x1_dict[word])

  f.close()
  return ordered_list, ordered_list_2

def prepare_lists(x1_dict,x1_bi_dict, intersection):
    x1_dict_new = []
    x1_bi_dict_new = []

    for word in intersection:
        x1_dict_new.append(x1_dict[word])
        x1_bi_dict_new.append(x1_bi_dict[word])

    return x1_dict_new, x1_bi_dict_new


def calculate_valNorm(file_name_1, file_name_2):
    x1_dict, vocab_list, x1 = read_vocab(file_name_1)
    x2,x1_prime = read_vocab2(file_name_2, x1_dict.keys(), x1_dict)
    count = min(len(x1), len(x2))
    print(len(x1_prime))
    print(len(x2))
    # calculate Pearson's correlation
    corr, _ = pearsonr(x1_prime, x2)
    print('Pearsons correlation for monolingual: %.3f' % corr)
