

class Compiler:
    """Class to compile the model with a specific optimizer, loss function and metrics."""
    def __init__(self, model, optimizer, loss_function, metrics):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=self.metrics)
