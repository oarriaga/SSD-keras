# taking from https://github.com/fchollet/keras/issues/1638
import threading


class threadsafe_iterator:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def threadsafe_generator(generator):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def wrapped_generator(*args, **kwargs):
        return threadsafe_iterator(generator(*args, **kwargs))
    return wrapped_generator
