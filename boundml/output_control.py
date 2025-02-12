import sys


class OutputControl:
    def __init__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.trash = open("/dev/null", "w")

    def mute(self):
        sys.stdout = self.trash
        sys.stderr = self.trash

    def unmute(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr

    def __del__(self):
        self.trash.close()
