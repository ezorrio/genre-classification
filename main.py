from MusicSearch import *

if __name__ == '__main__':
    search = MusicSearch("metadata/")
    search.train()
    search.test()
