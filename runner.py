"""Обробка цілої папки. Фото однієї зйомки освітлені однаково, тому колір надійніше
рахувати по серії, ніж по одному кадру. Якщо в папці кілька зйомок, вони визначаються
автоматично за різкою зміною освітлення між сусідніми (за іменем) фото."""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import enhance as E, imageio as IO

EXT = ('.jpg', '.jpeg', '.JPG', '.JPEG')


def files_in(src):
    return sorted(f for f in glob.glob(os.path.join(src, '*')) if f.endswith(EXT))


def _global_illum(sm, p=6):
    lin = E.s2l8(sm)
    mx, mn = E.chmax(lin), E.chmin(lin)
    v = (mx < 0.92) & (mn > 0.02)
    if v.sum() < 500: return np.ones(3, np.float32)
    e = (lin[v]**p).mean(0)**(1/p)
    return (e/e[1]).astype(np.float32)


def scan(path, rotate=0, long_side=1400):
    rgb, _ = IO.load(path, rotate, max_side=long_side)
    sm, faces = E.detect_faces(rgb, long_side)
    chroma, nf = E.skin_chroma(sm, faces)
    return dict(path=path, sm_size=(sm.shape[1], sm.shape[0]), faces=faces,
                chroma=chroma, nf=nf, illum=_global_illum(sm))


def segment(scans, jump=0.11, k=2.6):
    """Межі зйомок: там, де освітлення між сусідніми фото змінюється стрибком."""
    f = []
    for s in scans:
        a = np.log(s['chroma']) if s['chroma'] is not None else np.zeros(3, np.float32)
        f.append(np.concatenate([a[[0, 2]], np.log(s['illum'])[[0, 2]]*0.6]))
    f = np.array(f)
    if len(f) < 3: return [list(range(len(f)))]
    d = np.linalg.norm(np.diff(f, axis=0), axis=1)
    thr = max(jump, k*float(np.median(d)))
    segs, cur = [], [0]
    for i, v in enumerate(d, 1):
        if v > thr: segs.append(cur); cur = []
        cur.append(i)
    segs.append(cur)
    return [s for s in segs if s]


def series_cast(scans, idx, ref=E.REF_SKIN, min_n=3):
    cs = [E.cast_of(scans[i]['chroma'], ref) for i in idx if scans[i]['chroma'] is not None]
    if len(cs) < min_n: return None
    return tuple(float(v) for v in np.median(np.array(cs), 0))


def casts_for(paths, mode='auto', sample=48, log=None):
    """Колір по зйомках. Для великих папок достатньо вибірки кадрів."""
    if mode == 'photo': return {p: None for p in paths}
    step = max(1, len(paths)//sample)
    pick = list(range(0, len(paths), step))
    scans = [scan(paths[i]) for i in pick]
    out = {}
    segs = segment(scans)
    if log: log('Зйомок у папці: %d' % len(segs))
    for si, seg in enumerate(segs):
        c = series_cast(scans, seg)
        lo = pick[seg[0]]; hi = pick[seg[-1]]+step if seg[-1]+1 < len(pick) else len(paths)
        for i in range(lo, hi): out[paths[i]] = c
        if log and c: log('  зйомка %d: температура %+.3f, тінт %+.3f' % (si+1, c[0], c[1]))
    for p in paths: out.setdefault(p, None)
    return out


def process_one(path, dst, cast=None, zones=None, adjust=None, rotate=0, p=E.P, ref=E.REF_SKIN):
    rgb, meta = IO.load(path, rotate)
    r = E.with_adjust(E.analyze(rgb, ref, p, series_cast=cast), adjust)
    out = E.render(rgb, r, p, zones)
    if dst: IO.save(out, dst, meta)
    r['file'] = os.path.basename(path)
    return {k: v for k, v in r.items() if k not in ('face_fix', 'face_box', 'hist')}


def run(src, dst, mode='auto', settings=None, log=print):
    files = files_in(src); os.makedirs(dst, exist_ok=True)
    log('Фото в папці: %d' % len(files))
    cast = casts_for(files, mode, log=log)
    rep = []
    for i, f in enumerate(files):
        st = (settings or {}).get(os.path.basename(f)) or {}
        r = process_one(f, os.path.join(dst, os.path.basename(f)), cast.get(f),
                        st.get('zones'), st.get('adjust'), st.get('rotate', 0))
        rep.append(r)
        log('  %s  темп %+.3f  тінт %+.3f  EV %+.2f  гамма %.3f  контраст %.2f  насиченість %.2f'
            % (r['file'], r['temp'], r['tint'], r['ev'], r['gamma'], r['contrast'], r['sat']))
    json.dump(rep, open(os.path.join(dst, 'report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    return rep


def work(a):
    """Одне фото для паралельної обробки (має бути у звичайному модулі,
    інакше не працює у зібраному .exe)."""
    path, dst, cast, zones, adjust, rotate = a
    try:
        process_one(path, dst, cast, zones, adjust, rotate)
        return None
    except Exception:
        import traceback
        return '%s: %s' % (os.path.basename(path), traceback.format_exc(limit=1))


if __name__ == '__main__':
    a = sys.argv[1:]
    run(a[0], a[1], mode=a[a.index('--mode')+1] if '--mode' in a else 'auto')
