import logging
import torch
from datetime import datetime


def print_args(args):
    logging.info("----------Arguments-----------")
    for arg in vars(args):
        logging.info(str(arg)+' = '+str(getattr(args, arg)))
    logging.info("------------------------------\n")


def set_logging(args):
    log_filename = args.log_path + '/' + args.dataset.split('/')[0] + '_model_' + args.batch + '_' + datetime.now().strftime(
        "%m%d-%H%M") + '.log'
    logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("\x1b[38;20m" + ' %(message)s' + "\x1b[0m"))
    logging.getLogger().addHandler(console)


def set_device(args):
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        logging.info("\n no gpu found, program is running on cpu! \n")
    return device


