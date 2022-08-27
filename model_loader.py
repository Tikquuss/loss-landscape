def load(lightning_module_class, model_file):
    return lightning_module_class.load_from_checkpoint(model_file)