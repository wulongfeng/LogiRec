
# Gift_Cards

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/Gift_Cards -n 128 -b 512 -d 400 -g 60 \
  #-lr 0.0001 --max_steps 200000 --cpu_num 6 --geo beta --valid_steps 20000 \
  -lr 0.0001 --max_steps 200000 --cpu_num 6 --valid_steps 20000 \
  --save_checkpoint_steps 20000


#CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
#  --data_path data/Gift_Cards/lowOrder -n 128 -b 512 -d 400 -g 60 \
#  -lr 0.0001 --max_steps 200000 --cpu_num 6 --valid_steps 20000 \
#  --save_checkpoint_steps 20000


#CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
#  --data_path data/Gift_Cards/highOrder -n 128 -b 512 -d 400 -g 60 \
#  -lr 0.0001 --max_steps 200000 --cpu_num 6 --valid_steps 20000 \
#  --save_checkpoint_steps 20000
