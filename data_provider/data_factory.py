from data_provider.data_loader import airfoil, ns, darcy, pipe, elas, plas

def get_data(args):
    data_dict = {
        'elas': elas,
        'pipe': pipe,
        'airfoil': airfoil,
        'darcy': darcy,
        'ns': ns,
        'plas': plas,
    }
    dataset = data_dict[args.loader](args)
    train_loader, test_loader, shapelist = dataset.get_loader()
    return dataset, train_loader, test_loader, shapelist