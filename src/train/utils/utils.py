import yaml

def process_config_file(config_file, sanity_check_fn=None):
    # load config file
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    # run sanity check function
    if sanity_check_fn:
        sanity_check_fn(config)
    
    return config
