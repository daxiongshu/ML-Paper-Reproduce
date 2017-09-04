
def run_task(flags):
    task = flags.task
    if task == "test_one_image":
        if flags.net == "densenet121":
            from papers.densenet.densenet121 import densenet121
            model = densenet121(flags)
        if flags.net == "densenet161":
            from papers.densenet.densenet161 import densenet161
            model = densenet161(flags)
        if flags.net == "densenet169":
            from papers.densenet.densenet169 import densenet169
            model = densenet169(flags)
        model.inference_one_image(flags.input_path)
