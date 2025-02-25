from models import Transolver, LSM, FNO, U_Net, Transformer


def get_model(args):
    model_dict = {
        'Transformer': Transformer,
        'U_Net': U_Net,
        'FNO': FNO,
        'Transolver': Transolver,
        'LSM': LSM,
    }
    return model_dict[args.model].Model(args)
