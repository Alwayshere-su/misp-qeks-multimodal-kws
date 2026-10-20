"""
解压 train.zip — 支持只解压部分类型 (wav/label/mp4/fa)
用法: python unzip_train.py wav label
"""
import zipfile, sys, time
from pathlib import Path
from tqdm import tqdm

ZIP = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛\train.zip")
DEST = Path(r"d:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛\train_data")


def extract_types(types):
    z = zipfile.ZipFile(ZIP)
    infos = [i for i in z.infolist()
             if not i.is_dir() and i.filename.split('/')[0] in types]
    print(f"extracting {len(infos)} files of types {types}")
    t0 = time.time()
    for i in tqdm(infos, desc=f"unzip {types}"):
        # 去掉顶层目录 (wav/xxx -> train_data/xxx)
        rel = '/'.join(i.filename.split('/')[1:])
        out = DEST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with z.open(i) as src, open(out, 'wb') as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
    print(f"done {types} in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    types = sys.argv[1:] or ['wav', 'label']
    extract_types(types)
