from modules.xtra import XTRA
from modules.cls import Classifier


def get_model(args):
    if args.module == 'classifier':  # finetune
        return Classifier(args)
    else:
        return XTRA(args)            # pre-train
