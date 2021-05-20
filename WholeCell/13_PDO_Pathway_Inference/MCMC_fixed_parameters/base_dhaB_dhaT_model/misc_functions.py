import pickle
import numpy as np

def transform_from_log_unif(log_params,bounds):
	params = {}
	for param_name, param_val in log_params.items():
		if param_name in bounds.keys():
			bound_a, bound_b = bounds[param_name]
			param_trans = (bound_b - bound_a)*param_val + bound_a 
			params[param_name] = 10**param_trans
		else:
			params[param_name] = param_val
	return params

def transform_to_log_unif(params,bounds):
	log_params = {}
	for param_name, param_val in params.items():
		if param_name in bounds.keys():
			bound_a,bound_b = bounds[param_name]
			log_params[param_name] = (np.log10(param_val) - bound_a)/(bound_b - bound_a)
		else:
			log_params[param_name] = param_val
	return log_params

def transform_from_log_norm(log_params):
	params = {}
	for param_name, param_val in log_params.items():
		params[param_name] = 10**param_val
	return params

def transform_to_log_norm(params):
	log_params = {}
	for param_name, param_val in params.items():
		log_params[param_name] = np.log10(param_val)
	return log_params

def load_obj(name):
    """
    Load a pickle file. Taken from
    https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
    :param name: Name of file
    :return: the file inside the pickle
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    """
    Save a pickle file. Taken from
    https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file

    :param  obj: object to save
            name: Name of file
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


