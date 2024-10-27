import wandb

class BaseLogger:
    def __init__(self, tb_writer, use_wandb=False):
        self.writer = tb_writer
        self.use_wandb = use_wandb

    def log(self, d_result, i):
        for key, val in d_result.items():
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
                if self.use_wandb:
                    wandb.log({key: val}, step=i)
            if key.endswith('@'):
                if val is not None:
                    self.writer.add_image(key, val, i)
                    if self.use_wandb:
                        img = wandb.Image(val)
                        wandb.log({key: img}, step=i)