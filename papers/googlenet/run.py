
def run_task(flags):
    task = flags.task
    if task == "translate":
        if flags.net == "inception_v3":
            from papers.googlenet.translate_weight import translate_inception_v3_from_keras_app
            translate_inception_v3_from_keras_app()
    if task == "test_one_image":
        if flags.net == "inception_v3":
            from papers.googlenet.inception_v3 import inception_v3
            model = inception_v3(flags)
            model.inference_one_image(flags.input_path)
