class CsvLog:

    '''csv log writer to save all the parameters and metrics during the training'''

    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.line = []

    def add2line(self, l):
        if self.filename is not None:
            self.line += list(map(str, l))

    def writeline(self):
        if self.filename is not None:
            prefix = "\n"
            if self.file is None:
                self.file = open(self.filename, 'w')
                prefix = ""
            self.file.write(prefix + ",".join(self.line))
            self.file.flush()
            self.line = []

    def close(self):
        if self.file is not None:# and self.filename is not None::
            self.file.close()
