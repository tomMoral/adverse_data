

def repeat(n_rep):
    def decorator(func):
        func._repeat = n_rep
        return func
    return decorator
