import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def print_args(args = None):
    dic = vars(args)
    print()
    keys = sorted(dic.keys())
    print("common flags:")
    for i in keys:
        if dic[i]:
            print('{:>23} {:>23}'.format(i, str(dic[i])))
    print()


def main(FLAGS):
    print_args(FLAGS)
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
    from flags import get_parser
    parser = get_parser()
    args = parser.parse_args() 
    FLAGS = args
    main(FLAGS)
