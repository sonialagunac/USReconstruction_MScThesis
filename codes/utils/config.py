"""
Mélanie Bernhardt - M.Sc. Thesis - ETH Zurich
Utils file to convert the yaml file to a Bunch object
and initializing all configuration parameters.
"""
from bunch import Bunch
import yaml


def getConfig(file):
    """
    Get the config from a yaml file. 
    Adds the default value for missing parameters
    in the configuration file.

    Args:
        file: yaml file containing the configuration
    Returns:
        config: Bunch object. 
    """
    # parse the configurations from the config json file provided
    with open(file, 'r') as config_file:
        config_dict = yaml.load(config_file)
    print(config_dict)
    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)
    try:
        if config.c_init:
            pass
    except AttributeError:
        config.c_init = 1510

    try:
        if config.save_matrices_val:
            pass
    except AttributeError:
        config.save_matrices_val = True

    try:
        if config.freeze:
            pass
    except AttributeError:
        config.freeze = False

    try:
        if config.use_reg_activation_reg:
            pass
    except AttributeError:
        config.use_reg_activation_reg = False

    try:
        if config.use_reg_activation_data:
            pass
    except AttributeError:
        config.use_reg_activation_data = False

    try:
        if config.lambda_reg:
            pass
    except AttributeError:
        config.lambda_reg = 1e6

    try:
        if config.lambda_reg_data:
            pass
    except AttributeError:
        config.lambda_reg_data = 1e6

    try:
        if config.share_weights:
            pass
    except AttributeError:
        config.share_weights = False

    try:
        if config.mix:
            try:
                if config.filename_mix:
                    pass
            except AttributeError:
                raise('Specify file for mix')
            try:
                if config.mix_type:
                    pass
            except AttributeError:
                config.mix_type = 'syn'
    except AttributeError:
        config.mix = False
        config.filename_mix = None
        config.mix_type = None

    try:
        if config.p_mix:
            pass
    except AttributeError:
        config.p_mix = 1

    try:
        if config.p_triple:
            pass
    except AttributeError:
        config.p_triple = 1

    try:
        if config.mix_triple:
            try:
                if config.filename_mix_triple:
                    pass
            except AttributeError:
                raise('Specify file for triple mix')
            try:
                if config.mix_type_triple:
                    pass
            except AttributeError:
                config.mix_type_triple = 'ideal_time'

    except AttributeError:
        config.mix_triple = False
        config.mix_type_triple = None
        config.filename_mix_triple = None
    config.data_term_type = 'adaptive'
    config.use_preconditioner = True
    config.use_interp_reg_act = True
    config.inpaint_method = 'mean'
    if config.inpaint_method == 'median':
        config.use_med_filter = True
        config.use_mean_filter = False
    elif config.inpaint_method == 'mean':
        config.use_med_filter = False
        config.use_mean_filter = True
    return config
