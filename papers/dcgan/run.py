def run_task(flags):
    task = flags.task
    from papers.dcgan.dcgan import DCGAN
    model = DCGAN(flags)
    if task == "train":
        model.train()
    if task == "generate":
        model.gen()
