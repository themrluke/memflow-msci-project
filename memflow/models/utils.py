def lowercase_recursive(obj):
    if isinstance(obj,str):
        return obj.lower()
    elif isinstance(obj,(float,int)):
        return obj
    elif isinstance(obj,list):
        return [lowercase_recursive(element) for element in obj]
    elif isinstance(obj,tuple):
        return tuple(lowercase_recursive(list(obj)))
    elif isinstance(obj,dict):
        return {lowercase_recursive(key):lowercase_recursive(val) for key,val in obj.items()}
    else:
        raise TypeError(f'Type {type(obj)} not implemented')
