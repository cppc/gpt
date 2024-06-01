from abc import ABC


class Context(ABC):
    def __init__(self, *args):
        d = {}
        for a in args:
            d.update(a)
        self.vals = d

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, item):
        return self.vals[item]

    def __setitem__(self, key, value):
        self.vals[key] = value

    def __call__(self, *args, **kwargs):
        return self.vals[args[0]]

    def __getattr__(self, item):
        return self.vals[item]

    def __delitem__(self, key):
        del self.vals[key]

    def __delattr__(self, item):
        del self.vals[item]

    def __contains__(self, item):
        return item in self.vals

    def __iter__(self):
        x = self.vals.keys()
        return x.__iter__()

    def update(self, d):
        for k in d:
            self.vals[k] = d[k]

    def __str__(self):
        return self.vals.__str__()
