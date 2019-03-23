import blosc


class Compressor:
    def __init__(self, method, compress_level):
        self.method = method
        self.clevel = compress_level

    def compress_bytes(self, str_bytes):
        return blosc.compress(str_bytes, clevel=self.clevel, cname=self.method)
