rem Translate pretrained weights, pretrain_weights\inception_v3.npy will be generated
rem python main.py --paper googlenet --task translate --net inception_v3

rem Inference/test one image with pretrained model
python main.py --paper googlenet --task test_one_image --net inception_v3 --input_path data\images\cat.jpg --load_path pretrain_weights\inception_v3.npy
