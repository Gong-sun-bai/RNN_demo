#!/bin/bash

# 通知执行的命令
echo "Starting execution of RNN.py"
/home/ps/anaconda3/envs/changemamba_lwb/bin/python /home/gdut_students/lwb/RNN_network/RNN.py
echo "Finished execution of RNN.py"

echo "Starting execution of LSTM.py"
/home/ps/anaconda3/envs/changemamba_lwb/bin/python /home/gdut_students/lwb/RNN_network/LSTM.py
echo "Finished execution of LSTM.py"

echo "Starting execution of GRU.py"
/home/ps/anaconda3/envs/changemamba_lwb/bin/python /home/gdut_students/lwb/RNN_network/GRU.py
echo "Finished execution of GRU.py"

echo "Starting execution of MLP.py"
/home/ps/anaconda3/envs/changemamba_lwb/bin/python /home/gdut_students/lwb/RNN_network/MLP.py
echo "Finished execution of MLP.py"

echo "All tasks completed!"
