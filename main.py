import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
FLAGS = None

def print_args():
    dic = FLAGS.__flags#vars(args)
    print()
    keys = sorted(dic.keys())
    print("Used flags:")
    for i in keys:
        if dic[i]:
            print('{:>23} {:>23}'.format(i, str(dic[i])))
    print()


def main(_):
    print_args()
    if FLAGS.paper == "resnet":
        from papers.resnet.run import run_task
    elif FLAGS.paper == "googlenet":
        from papers.googlenet.run import run_task
    elif FLAGS.paper == "densenet":
        from papers.densenet.run import run_task
    else:
        print("Unknown paper name %s"%FLAGS.paper)
        assert False
    print("run Paper: %s Task: %s"%(FLAGS.paper,FLAGS.task))
    run_task(FLAGS)

if __name__ == "__main__":
    from flags import FLAGS
    tf.app.run()
