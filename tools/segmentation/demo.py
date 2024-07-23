# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmfewshot.mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmfewshot.mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='isaid',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img)

    import cv2
    import numpy as np
    import pdb
    pdb.set_trace()
    cv2.imwrite('result.png', np.where(result[0]>0, 255, 0).astype(np.uint8))
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        get_palette(args.palette),
        opacity=args.opacity)


if __name__ == '__main__':
    main()
