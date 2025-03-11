from models import Transolver, LSM, FNO, U_Net, Transformer, Factformer, Swin_Transformer, Galerkin_Transformer, GNOT, \
    U_NO, U_FNO, F_FNO, ONO, MWT, GraphSAGE, Graph_UNet, PointNet


def get_model(args):
    model_dict = {
        'PointNet': PointNet,
        'Graph_UNet': Graph_UNet,
        'GraphSAGE': GraphSAGE,
        'MWT': MWT,
        'ONO': ONO,
        'F_FNO': F_FNO,
        'U_FNO': U_FNO,
        'U_NO': U_NO,
        'GNOT': GNOT,
        'Galerkin_Transformer': Galerkin_Transformer,
        'Swin_Transformer': Swin_Transformer,
        'Factformer': Factformer,
        'Transformer': Transformer,
        'U_Net': U_Net,
        'FNO': FNO,
        'Transolver': Transolver,
        'LSM': LSM,
    }
    return model_dict[args.model].Model(args)
