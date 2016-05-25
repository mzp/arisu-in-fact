from learn import infer
import shutil
import sys

if __name__ == '__main__':
    for x in sys.argv[1:]:
        pred = infer('model', 2, x)

        if pred == 0:
            shutil.copy(x, 'arisu.infer')
        else:
            shutil.copy(x, 'not-arisu.infer')

