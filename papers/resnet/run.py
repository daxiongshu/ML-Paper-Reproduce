
def run_task(flags):
    task = flags.task
    if task == "translate":
        if flags.net == "resnet50":
            from papers.resnet.translate_weight import translate_resnet50_from_keras_app
            translate_resnet50_from_keras_app()
    if task == "test_one_image":
        if flags.net == "resnet50":
            from papers.resnet.resnet50 import resnet50
            model = resnet50(flags)
            model.inference_one_image(flags.input_path)
