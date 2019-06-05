class FinishSetupNotCalledInConstructor(Exception):
    """a class that inherits from BaseModel forgot to call self.finish_setup() at the end of __init__ """
    pass
