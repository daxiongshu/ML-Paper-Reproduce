
def run_task(flags):
    task = flags.task
    if task == "test_one_image":
        if flags.net == "vgg16":
            from papers.vgg.vgg16 import vgg16
            model = vgg16(flags)
            model.inference_one_image(flags.input_path)
