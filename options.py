import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-whattodo', type=int, default=1, help='1-train ner and re models (default), 2-use existing models to extract entities and relations from raw text')
parser.add_argument('-config', default='default.config', help='configuration file (default=default.config)')
parser.add_argument('-output', default='./output', help='output directory of whattodo 1 (default=./output)')
parser.add_argument('-pretrained_model_dir', default='None', help='directory of pretrained models (default=None)')
parser.add_argument('-input', default='./input', help='input directory of whattodo 2 (default=./input)')
parser.add_argument('-predict', default='./predict', help='output directory of whattodo 2 (default=./predict)')
parser.add_argument('-test_in_cpu', action='store_true', default=False, help="add this flag if you don't use gpu (default=False)")
parser.add_argument('-verbose', action='store_true', help='add this flag to print debug logs (default=False)')
parser.add_argument('-gpu', type=int, default=0, help='use which gpu for whattodo 1 (default=0)')
parser.add_argument('-old_gpu', type=int, default=0, help='use which gpu for whattodo 2 (default=0)')

opt = parser.parse_args()

