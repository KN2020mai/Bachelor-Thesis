```python
    parser.add_argument('--model', type=str, default='cpnet', choices=['cpnet', 'unet', 'dresunet'], help='model name (default: resnet)')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101'], help='backbone network for dresunet (default: resnet50)')
    parser.add_argument('--use-tmap', action='store_true', default=False, help='whether to use tmap (default: auto)')
    parser.add_argument('--tmap', type=str, default='weight', choices=['weight', 'output'], help='tmap used in loss weight or output (default: weight)')
    parser.add_argument('--use-sam', action='store_true', default=False, help='whether to use spatial attention module (default: auto)')
    parser.add_argument('--sam-position', type=str, default='mid', choices=['mid', 'tail'], help='spatial attention module used in mid or tail (default: mid)')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--nepochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--diam-mean', type=float, default=17.0, help='mean of the diameter')
```

python train_sartorius.py --model cpnet --use-tmap --tmap output --use-sam --sam-position mid --nepochs 5
python train_sartorius.py --model unet --use-tmap --tmap output --use-sam --sam-position mid
python train_sartorius.py --model dresunet --backbone resnet50 --use-tmap --tmap output --use-sam --sam-position mid