"""整理: 移动有依赖的脚本到 scripts/ 子目录, 并修复 sys.path 指向项目根"""
from pathlib import Path

BASE = Path(r"D:\MMLL\4.competition\2026多模态开放词汇关键词检索挑战赛")

moves = {
    'make_submission.py': 'scripts/submission/',
    'make_submission_p1.py': 'scripts/submission/',
    'make_submission_p3.py': 'scripts/submission/',
    'make_submission_p4.py': 'scripts/submission/',
    'make_ensemble_submissions.py': 'scripts/submission/',
    'fuse_eval.py': 'scripts/submission/',
    'fuse_p1p4_s2u.py': 'scripts/submission/',
    'score_val_acoustic.py': 'scripts/submission/',
    'score_val_oof.py': 'scripts/submission/',
    'score_val_s2u.py': 'scripts/submission/',
    'extract_lip_feats.py': 'scripts/features/',
    'extract_train_feats.py': 'scripts/features/',
    'extract_whisper_feats.py': 'scripts/features/',
    'eval_asr_quality.py': 'scripts/asr/',
    'test_asr.py': 'scripts/asr/',
    'save_val_asr.py': 'scripts/asr/',
    'test_asr_stats.py': 'scripts/asr/',
    'check_feature_shift.py': 'scripts/analysis/',
    'verify_frame_match.py': 'scripts/analysis/',
}

for f, dst in moves.items():
    src = BASE / f
    if not src.exists():
        print(f'skip {f}')
        continue
    (BASE / dst).mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding='utf-8', errors='ignore')
    # 修复 sys.path: scripts/子目录 的 parent.parent = 项目根
    new = text.replace("sys.path.insert(0, str(Path(__file__).parent))",
                       "sys.path.insert(0, str(Path(__file__).parent.parent))")
    new = new.replace("sys.path.insert(0, str(Path(__file__).parent.resolve()))",
                      "sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))")
    # 若完全无 sys.path 但依赖项目模块 -> 插入
    if 'sys.path' not in new:
        new = ('import sys\nfrom pathlib import Path\n'
               'sys.path.insert(0, str(Path(__file__).parent.parent))\n' + new)
    (BASE / dst / f).write_text(new, encoding='utf-8')
    src.unlink()
    print(f'moved {f} -> {dst}')

print('完成')
