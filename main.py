#!/usr/bin/env python
from lightning.pytorch.cli import LightningCLI

def main():
    cli = LightningCLI(save_config_kwargs={'overwrite': True})

if __name__ == '__main__':
    main()