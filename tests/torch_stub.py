class Tensor:
    def __init__(self, shape):
        self._shape = tuple(shape)

    @property
    def shape(self):
        return self._shape

    def mean(self, dim=None):
        if dim is None:
            self._shape = (1,)
        else:
            if isinstance(dim, int):
                dim = (dim,)
            dims = sorted([(d if d >= 0 else len(self._shape) + d) for d in dim], reverse=True)
            shape = list(self._shape)
            for d in dims:
                shape.pop(d)
            if not shape:
                shape = (1,)
            self._shape = tuple(shape)
        return self

    def repeat(self, *reps):
        if len(reps) == 1 and isinstance(reps[0], tuple):
            reps = reps[0]
        self._shape = tuple(s * r for s, r in zip(self._shape, reps))
        return self

    def __getitem__(self, item):
        if isinstance(item, tuple) and isinstance(item[1], slice):
            stop = item[1].stop if item[1].stop is not None else self._shape[1]
            start = item[1].start or 0
            c = stop - start
            shape = list(self._shape)
            shape[1] = c
            self._shape = tuple(shape)
        return self


def randn(*shape):
    return Tensor(shape)
