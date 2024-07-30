"""
command-line argument parsing utilities

Example:

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Config file')
parser.add_argument('--dataset', choices=['8gaussians', 'checkerboard', 'swissroll'], default='8gaussians', help='Dataset to use')
parser.add_argument('--run', default=None, type=str, help='Run name')
parser.add_argument('--device', default=0, type=str, help='Device to use')
args, unknown = parser.parse_known_args()
d_cmd_cfg = parse_unknown_args(unknown)
d_cmd_cfg = parse_nested_args(d_cmd_cfg)
"""

def parse_arg_type(val):
    if val.isnumeric():
        return int(val)
    try:
        return float(val)
    except ValueError:

        if val.lower() == 'true':
            return True
        elif val.lower() == 'false':
            return False
        elif val.lower() == 'null' or val.lower() == 'none':
            return None
        elif val.startswith('[') and val.endswith(']'):  # parse list
            return eval(val)
        return val


def parse_unknown_args(l_args):
    """convert the list of unknown args into dict
    this does similar stuff to OmegaConf.from_cli()
    I may have invented the wheel again..."""
    n_args = len(l_args) // 2
    kwargs = {}
    for i_args in range(n_args):
        key = l_args[i_args*2]
        val = l_args[i_args*2 + 1]
        assert '=' not in key, 'optional arguments should be separated by space'
        kwargs[key.strip('-')] = parse_arg_type(val)
    return kwargs


def parse_nested_args(d_cmd_cfg):
    """produce a nested dictionary by parsing dot-separated keys
    e.g. {key1.key2 : 1}  --> {key1: {key2: 1}}"""
    d_new_cfg = {}
    for key, val in d_cmd_cfg.items():
        l_key = key.split('.')
        d = d_new_cfg
        for i_key, each_key in enumerate(l_key):
            if i_key == len(l_key) - 1:
                d[each_key] = val
            else:
                if each_key not in d:
                    d[each_key] = {}
                d = d[each_key]
    return d_new_cfg