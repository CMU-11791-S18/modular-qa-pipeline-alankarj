import pickle
import json
import collections


def main():
    y_true = pickle.load(open('true.pkl', 'rb'))
    y_nb_c = pickle.load(open('nb_c.pkl', 'rb'))
    y_nb_t = pickle.load(open('nb_t.pkl', 'rb'))
    y_mlp_c = pickle.load(open('mlp_c.pkl', 'rb'))
    y_mlp_t = pickle.load(open('mlp_t.pkl', 'rb'))

    num_incorrect = dict()
    num_incorrect_reverse = dict()

    for i, y in enumerate(y_true):
        num_incorrect[i] = 0

    for i in range(7):
        num_incorrect_reverse[i] = 0

    for i, y in enumerate(y_true):
        if y_nb_c[i] != y:
            num_incorrect[i] += 1
        if y_nb_t[i] != y:
            num_incorrect[i] += 1
        if y_mlp_c[i] != y:
            num_incorrect[i] += 1
        if y_mlp_t[i] != y:
            num_incorrect[i] += 1

    for k, v in num_incorrect.items():
        num_incorrect_reverse[v] += 1

    for k, v in num_incorrect_reverse.items():
        num_incorrect_reverse[k] = float(num_incorrect_reverse[k]/len(y_true))*100

    # print(json.dumps(num_incorrect, indent=2))
    print(json.dumps(num_incorrect_reverse, indent=2))

    num_incorrect = dict()
    counter = dict()

    for i, y in enumerate(y_true):
        num_incorrect[y] = 0
        counter[y] = 0

    for i, y in enumerate(y_true):
        counter[y] += 1
        if y_nb_c[i] != y:
            num_incorrect[y] += 1
        if y_nb_t[i] != y:
            num_incorrect[y] += 1
        if y_nb_c[i] != y:
            num_incorrect[y] += 1
        if y_nb_t[i] != y:
            num_incorrect[y] += 1

    for k, v in num_incorrect.items():
        num_incorrect[k] = float(num_incorrect[k] / (4 * counter[k]))

    # print(json.dumps(num_incorrect, indent=2))
    # print(json.dumps(counter, indent=2))

    counter_tuple = sorted([(k, v) for k, v in counter.items()], reverse=True, key=lambda x: x[1])

    print(len(counter_tuple))

    for k, v in counter_tuple[:10]:
        print(k, ' ', v, end='')
        print(' %.2f' % num_incorrect[k])

    pairwise = dict()
    for i in range(4):
        pairwise[i] = []

    for i, y in enumerate(y_true):
        if (y_nb_c[i] != y) and (y_mlp_t[i] != y):
            pairwise[0].append(y)
        elif (y_nb_c[i] == y) and (y_mlp_t[i] == y):
            pairwise[1].append(y)
        elif (y_nb_c[i] != y) and (y_mlp_t[i] == y):
            pairwise[2].append(y)
        elif (y_nb_c[i] == y) and (y_mlp_t[i] != y):
            pairwise[3].append(y)

    for i in range(4):
        x = collections.Counter(pairwise[i]).most_common(10)
        print(list(map(list, zip(*x))))


if __name__ == '__main__':
    main()