import sys


def get_graph(query):
    return {'a': ['b', 'c', 'd'],
            'b': ['a', 'c']}

if __name__ == '__main__':
    arg = sys.argv
    if arg[1] == 'get_graph':
        print(get_graph(arg[2]))