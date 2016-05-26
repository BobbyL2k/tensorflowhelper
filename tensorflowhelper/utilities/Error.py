class TFHError(Exception):
    def __init__(self, *message):
        self.message = message

    def __str__(self):
        return "TensorFlow Helper Error : " + repr(self.message)