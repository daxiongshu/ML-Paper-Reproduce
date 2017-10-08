def run_task(flags):
    task = flags.task
    if task == "train":
        from papers.dcgan.dcgan import DCGAN 
        model = DCGAN(flags)
        model.train()
