rem Inference/test one image with pretrained model
python main.py --paper densenet --task test_one_image --net densenet121 --input_path data\images\cat.jpg --load_path pretrain_weights\densenet121.npy
rem
python main.py --paper densenet --task test_one_image --net densenet161 --input_path data\images\cat.jpg --load_path pretrain_weights\densenet161.npy
rem
python main.py --paper densenet --task test_one_image --net densenet169 --input_path data\images\cat.jpg --load_path pretrain_weights\densenet169.npy
