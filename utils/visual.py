import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')


def visual(x, y, out, args, id):
    if args.geotype == 'structured_2D':
        visual_structured_2d(x, y, out, args, id)
    if args.geotype == 'unstructured' and x.shape[-1] == 2:
        visual_unstructured_2d(x, y, out, args, id)


def visual_unstructured_2d(x, y, out, args, id):
    plt.axis('off')
    plt.scatter(x=x[0, :, 0].detach().cpu().numpy(), y=x[0, :, 1].detach().cpu().numpy(),
                c=y[0, :].detach().cpu().numpy(), cmap='coolwarm')
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "gt_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.axis('off')
    plt.scatter(x=x[0, :, 0].detach().cpu().numpy(), y=x[0, :, 1].detach().cpu().numpy(),
                c=out[0, :].detach().cpu().numpy(), cmap='coolwarm')
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "pred_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.axis('off')
    plt.scatter(x=x[0, :, 0].detach().cpu().numpy(), y=x[0, :, 1].detach().cpu().numpy(),
                c=((y[0, :] - out[0, :])).detach().cpu().numpy(), cmap='coolwarm')
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "error_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()


def visual_unstructured_3d(x, y, out, args, id):
    pass


def visual_structured_1d(x, y, out, args, id):
    pass


def visual_structured_2d(x, y, out, args, id):
    if args.vis_bound is not None:
        space_x_min = args.vis_bound[0]
        space_x_max = args.vis_bound[1]
        space_y_min = args.vis_bound[2]
        space_y_max = args.vis_bound[3]
    else:
        space_x_min = 0
        space_x_max = args.shapelist[0]
        space_y_min = 0
        space_y_max = args.shapelist[1]
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   np.zeros([args.shapelist[0], args.shapelist[1]])[space_x_min: space_x_max, space_y_min: space_y_max],
                   shading='auto',
                   edgecolors='black', linewidths=0.1)
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "input_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   out[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   shading='auto', cmap='coolwarm')
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "pred_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   y[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   shading='auto', cmap='coolwarm')
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "gt_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   out[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy() - \
                   y[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   shading='auto', cmap='coolwarm')
    plt.colorbar()
    plt.savefig(
        os.path.join('./results/' + args.save_name + '/',
                     "error_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.close()


def visual_structured_3d(x, y, out, args, id):
    pass
