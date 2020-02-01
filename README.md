# DQN2048
the game is modified from https://github.com/ncchen99/2048ByNCC
a bug has been fixed: in case there are more than 3 same blocks in the moving direction, the first two (in the moving direction) should be combined first.

the train.py is the tensorflow keras version. while the train_pt.py is the pytorch version. And lastly, train_conv_pt.py is the pytorch GPU version.

At the begining, the game is in manual mode. "jkli" moves the blocks.
to enter auto mode, one should keep pressing 'a' key just after 'jkli' keys.
press q to quit.
