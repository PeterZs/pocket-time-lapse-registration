
from functools import wraps # This convenience func preserves name and docstring

def AddMethod(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator

def AddStaticMethod(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs);
        setattr(cls, func.__name__, staticmethod(wrapper))
        return func # returning func means func can still be used normally
    return decorator

def AddClassMethod(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs);
        setattr(cls, func.__name__, classmethod(wrapper))
        return func # returning func means func can still be used normally
    return decorator

# def AddProperty(cls):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(self, *args, **kwargs):
#             return func(*args, **kwargs)
#         setattr(cls, func.__name__, wrapper)
#         should be something like x = property(getx, setx, delx, "I'm the 'x' property.")
#         # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
#         return func # returning func means func can still be used normally
#     return decorator