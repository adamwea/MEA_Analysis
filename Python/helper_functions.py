import json
import numpy as np




def convert_int64_keys_to_ints(d):
    """
    Recursively convert all numpy.int64 keys in a dictionary to integers.
    """
    new_dict = {}
    for k, v in d.items():
        if isinstance(k, np.int64):
            k = int(k)
        if isinstance(v, dict):
            v = convert_int64_keys_to_ints(v)
        new_dict[k] = v
    return new_dict

def load_json(filename):
    d = {}
    with open(filename,'rb') as f:
        d = json.load(f)
    return d
def dumpdicttofile(data,filename):
    data = convert_int64_keys_to_ints(data)
    json_data = json.dumps(data,indent=4)

    with open(filename,'w') as fileptr:
        fileptr.write(json_data)

def get_key_by_value(d, value):
    """
    Useful to get the templates associated with a extreme electrode
    """
    keys = [k for k, v in d.items() if v == value]
    return keys if keys else None

def load_json(filename):
    d = {}
    with open(filename,'rb') as f:
        d = json.load(f)
    return d

def inarrxnotinarry(arr1,arr2):
    print(f" in array1 not in array2 :{[x for x in arr1 if x not in arr2]} ")
    print(f" in array2 not in array1 :{[x for x in arr2 if x not in arr1]} ")


def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def get_templates_with_same_channels(electrode_file):

    my_dict = load_json(electrode_file)

    templates_with_channel = {}

    for template_name, template_data in my_dict.items():
        for channel_name in template_data:
            if channel_name in templates_with_channel:
                templates_with_channel[channel_name].append(template_name)
            else:
                templates_with_channel[channel_name] = [template_name]
    
    same_channel_templates = [templates for channel, templates in templates_with_channel.items() if len(templates) > 1]

    return same_channel_templates
