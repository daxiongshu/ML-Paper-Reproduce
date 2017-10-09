rem generate using pretrained weights. Please download weights at https://drive.google.com/open?id=0B67xNMmgTciJUXpzSExlVGlSd1U
python main.py --paper dcgan --task generate --load_path pretrain_weights/dcgan_None_None_19.npy --width 96 --height 96 --out_width 96 --out_height 96 --batch_size 64 --data_path papers\dcgan\data 

rem train from scratch. Please modify input_path, data_path and save_path
python main.py --paper dcgan --task train --input_path C:\Users\Jiwei\Documents\GitHub\DCGAN-tensorflow\data\anime --save_path pretrain_weights --width 96 --height 96 --out_width 96 --out_height 96 --batch_size 64 --opt adam --learning_rate 0.0002 --epochs 100 --verbosity 10 --data_path papers\dcgan\data --g_num_update 2

rem train with pretrained weights
python main.py --paper dcgan --task train --input_path C:\Users\Jiwei\Documents\GitHub\DCGAN-tensorflow\data\
anime --save_path pretrain_weights --width 96 --height 96 --out_width 96 --out_height 96 --batch_size 64 --opt adam --learning_rate 0.0002 --epochs 100 --verbosity 10 --data_path papers\dcgan\data --g_num_update 2 --load_path pretrain_weights/dcgan_None_None_19.npy --pre_epochs 19
