rem Translate pretrained weights, pretrain_weights\resnet50.npy will be generated
rem python main.py --paper resnet --task translate --net resnet50

rem Inference/test one image with pretrained model
python main.py --paper googlenet --task test_one_image --net inception_v3 --input_path data\images\cat.jpg 
rem --load_path pretrain_weights\resnet50.npy
