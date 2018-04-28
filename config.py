import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAINED_PATH = os.path.join(BASE_PATH, "semeval2018/trained")

EXPS_PATH = os.path.join(BASE_PATH, "semeval2018/out/experiments")

ATT_PATH = os.path.join(BASE_PATH, "semeval2018/out/attentions")
