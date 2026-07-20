class EMA:
  """Exponential Moving Average"""
  def __init__(self, model, decay = 0.999):
    self.model = model
    self.decay = decay
    self.shadow = {}
    self.backup = {}

    for name, param in model.named_parameters():
      if param.requires_grad:
        self.shadow[name] = param.data.clone()

  def update(self):
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        self.shadow[name].data = self.decay * self.shadow[name].data + (1 - self.decay) * param.data

  def apply_shadow(self):
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        self.backup[name] = param.data.clone()
        param.data = self.shadow[name].data

  def restore(self):
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        param.data = self.backup[name].data

  def state_dict(self):
    return self.shadow

  def load_state_dict(self, state_dict):
    self.shadow = state_dict
