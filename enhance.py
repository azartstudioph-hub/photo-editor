"""Автообробка фото: баланс білого (температура/тінт), експозиція, локальні тіні й світла,
чорна точка, контраст, насиченість, локальне освітлення облич та ручні зони.

Тон рахується в лінійному світлі. Локальні карти будуються на зменшеній копії й згладжуються
зі збереженням країв (guided filter), тому не виникає ореолів навколо контрастних країв.
"""
import os, sys, json
import numpy as np, cv2

def res_dir():
    """Папка з ресурсами: поряд зі скриптом або всередині зібраного .exe."""
    return getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))


HERE = res_dir()
YUNET = os.path.join(HERE, 'models', 'yunet2.onnx')
LUMW = np.array([0.2126, 0.7152, 0.0722], np.float32)

# Еталонна хроматичність шкіри (лінійні R/G, 1, B/G) при нейтральному світлі.
REF_SKIN = np.array([1.729, 1.0, 0.5430], np.float32)

P = dict(
    # колір
    cast_dead=0.010, cast_full=0.030, cast_strength=1.0, cast_cap=0.22,
    # експозиція (цільова світлота шкіри взята з еталонних прикладів)
    face_L=68.0, dead_lo=1.2, dead_hi=4.0, ev_div=13.0, ev_cap=0.9,
    face_ceil=0.86, cal_iter=5, cal_tol=0.7, cal_lo=66.5, cal_hi=73.5, cal_px=460,
    scene_med=0.55, scene_dead=0.05, scene_ev_cap=0.6,
    # Тіні й світла (EV). Смуга задана в звичних значеннях яскравості 0..1:
    # чорне вбрання (нижче sh_band[0]) не чіпаємо, піднімаємо лише напівтінь.
    sh_cap=0.75, sh_k=3.4, sh_lo=0.17, sh_hi=0.44, sh_band=(0.15, 0.26, 0.42, 0.62),
    hl_cap=0.9, hl_k=2.4, hl_ref=0.86, hl_band=(0.62, 0.95),
    # рівні: біла точка і середні тони (замість жорсткого S-контрасту)
    white_target=0.965, white_pull=0.45, white_cap=1.07, mid_target=0.675, mid_pull=0.6,
    gamma_lo=0.92, gamma_hi=1.08,
    black_target=0.030, black_cap=0.018, spread_target=0.78, contrast_cap=0.20,
    # насиченість і тепле підсвічування жовто-оранжевих тонів (шкіра, дерево, листя)
    sat_base=0.11, sat_target=0.46, sat_k=0.25, sat_cap=0.22, sat_band=(0.12, 0.92), skin_sat=0.10,
    skin_lum=0.03, skin_hue=(8.0, 18.0, 48.0, 64.0),
    # обличчя
    face_target=0.66, face_lift_k=1.3, face_lift_cap=0.55, face_lift_flat=0.30,
    face_shadow_k=1.7, face_shadow_cap=0.55,
    conf_n=2.0, photo_weight=0.45,
    knee=0.82, top_knee=0.95, map_scale=0.25, clip_budget=0.030, clip_level=251,
)

# ------------------------------------------------------------------ базові функції
def s2l(x): return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055)**2.4).astype(np.float32)

_S2L = s2l(np.arange(256, dtype=np.float32)/255.0)       # таблиця: uint8 -> лінійне


def s2l8(img8):
    """Те саме, що s2l, але з готової таблиці — у кілька разів швидше."""
    return _S2L[img8]


def chmax(a): return np.maximum(np.maximum(a[..., 0], a[..., 1]), a[..., 2])


def chmin(a): return np.minimum(np.minimum(a[..., 0], a[..., 1]), a[..., 2])

def l2s(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x*12.92, 1.055*np.power(x, 1/2.4)-0.055).astype(np.float32)

def luma(lin): return lin @ LUMW

def ss(x, a, b):
    """Плавний перехід від 0 до 1 між a і b (менше проміжних масивів — швидше)."""
    t = np.subtract(x, a, dtype=np.float32); t *= 1.0/(b-a+1e-9)
    np.clip(t, 0.0, 1.0, out=t)
    u = np.multiply(t, -2.0); u += 3.0; u *= t; u *= t
    return u

def _box(im, r): return cv2.boxFilter(im, -1, (2*r+1, 2*r+1), borderType=cv2.BORDER_REFLECT)

def guided(I, r, eps):
    mI = _box(I, r); var = np.maximum(_box(I*I, r)-mI*mI, 0)
    a = var/(var+eps); b = mI*(1-a)
    return _box(a, r)*I+_box(b, r)

def lightness(lin):
    """Світлота L* (0..100) лінійного кольору."""
    y = float(np.clip(np.asarray(lin, np.float64) @ LUMW, 0, 1))
    f = y**(1/3) if y > 0.008856 else 7.787*y+16/116
    return 116*f-16

# ------------------------------------------------------------------ обличчя і шкіра
def detect_faces(rgb8, long_side=1400, thr=0.70, min_px=22):
    h, w = rgb8.shape[:2]; s = long_side/max(h, w)
    sm = cv2.resize(rgb8, (max(int(w*s), 1), max(int(h*s), 1)), interpolation=cv2.INTER_AREA)
    det = cv2.FaceDetectorYN.create(YUNET, '', (sm.shape[1], sm.shape[0]), thr, 0.3, 500)
    _, f = det.detect(sm[:, :, ::-1].copy())
    faces = []
    if f is not None:
        for d in f:
            if d[14] < thr or d[2] < min_px: continue
            faces.append(dict(sbox=tuple(float(v) for v in d[:4]),
                              box=tuple(float(v)/s for v in d[:4]), score=float(d[14])))
    return sm, faces

def skin_pixels(sm, sbox, min_px=120):
    """Пікселі шкіри з центральної частини лиця. Спершу беремо лише нормально
    експоновані (темна чи пересвічена шкіра спотворює колір), за потреби розширюємо діапазон."""
    x, y, w, h = sbox
    x0, x1 = max(int(x+.22*w), 0), int(x+.78*w); y0, y1 = max(int(y+.32*h), 0), int(y+.84*h)
    roi = sm[y0:y1, x0:x1]
    if roi.size == 0: return np.zeros((0, 3), np.float32)
    ycc = cv2.cvtColor(roi[:, :, ::-1].copy(), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    base = ((ycc[:, 1] >= 133) & (ycc[:, 1] <= 175) & (ycc[:, 2] >= 76) & (ycc[:, 2] <= 128))
    flat = roi.reshape(-1, 3)
    for lo, hi in ((95, 220), (77, 235), (35, 248)):
        m = base & (ycc[:, 0] > lo) & (ycc[:, 0] < hi)
        if m.sum() >= min_px: return flat[m].astype(np.float32)/255.0
    return flat[m].astype(np.float32)/255.0

def skin_chroma(sm, faces, min_px=120):
    ms, ws = [], []
    for f in faces:
        sp = skin_pixels(sm, f['sbox'])
        if len(sp) >= min_px:
            m = np.median(s2l(sp), 0); ms.append(m/max(m[1], 1e-6)); ws.append(f['sbox'][2]**2)
    if not ms: return None, 0
    return np.exp(np.average(np.log(np.array(ms)), 0, weights=ws)).astype(np.float32), len(ms)

def skin_index(sm, faces, scale=1.0, min_px=80):
    """Координати пікселів шкіри на зображенні sm. За ними міряємо результат обробки."""
    ys, xs = [], []
    H, W = sm.shape[:2]
    for f in faces:
        x, y, w, h = (v*scale for v in f['sbox'])
        x0, x1 = max(int(x+.22*w), 0), min(int(x+.78*w), W)
        y0, y1 = max(int(y+.32*h), 0), min(int(y+.84*h), H)
        roi = sm[y0:y1, x0:x1]
        if roi.size == 0: continue
        ycc = cv2.cvtColor(roi[:, :, ::-1].copy(), cv2.COLOR_BGR2YCrCb)
        base = ((ycc[..., 1] >= 133) & (ycc[..., 1] <= 175)
                & (ycc[..., 2] >= 76) & (ycc[..., 2] <= 128))
        m = None
        for lo, hi in ((95, 220), (77, 235), (35, 248)):
            mm = base & (ycc[..., 0] > lo) & (ycc[..., 0] < hi)
            if mm.sum() >= min_px: m = mm; break
        if m is None: continue
        yy, xx = np.nonzero(m); ys.append(yy+y0); xs.append(xx+x0)
    if not ys: return None
    return np.concatenate(ys), np.concatenate(xs)


def skin_L(img8, idx):
    """Світлота шкіри L* на готовому зображенні."""
    return lightness(np.median(s2l8(img8[idx[0], idx[1]]), 0))


def cast_of(chroma, ref=REF_SKIN):
    """Відхилення світла: temp>0 — тепліше ніж треба, tint>0 — зеленіше."""
    e = chroma/ref; lr, lb = np.log(e[0]), np.log(e[2])
    return float((lr-lb)/2), float(-(lr+lb)/2)

def corr_gains(dt, dn):
    """dt>0 — тепліше, dn>0 — в бік пурпурного. Загальна яскравість не змінюється."""
    g = np.array([np.exp(dt), np.exp(-dn), np.exp(-dt)], np.float32)
    return g/float(g @ LUMW)

def soft(v, dead, full, strength, cap):
    w = float(np.clip((abs(v)-dead)/(full-dead), 0, 1))
    return float(np.clip(v*w*strength, -cap, cap))

# ------------------------------------------------------------------ локальні карти
def face_mask(shape, boxes, full_shape, grow=(0.75, 0.95)):
    hs, ws = shape; sy, sx = hs/full_shape[0], ws/full_shape[1]
    yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float32)
    out = np.zeros(shape, np.float32); each = []
    for b in boxes:
        x, y, w, h = b
        cx, cy = (x+w/2)*sx, (y+h/2)*sy
        rx, ry = max(w*grow[0]*sx, 2), max(h*grow[1]*sy, 2)
        d = np.sqrt(((xx-cx)/rx)**2+((yy-cy)/ry)**2)
        m = 1-ss(d, 0.55, 1.1); each.append(m); out = np.maximum(out, m)
    return out, each

def base_srgb(Y, p=P):
    """Великомасштабна яскравість (0..1) зі збереженням країв."""
    L = np.log2(np.maximum(Y, 1e-5))
    return l2s(2.0**guided(L, max(6, int(min(L.shape)*0.07)), 0.10))

def local_gain(Y, r, p, full_shape):
    """Карта підсилення в EV: тіні, світла, обличчя."""
    B = base_srgb(Y, p)
    a, b, c, d = p['sh_band']
    g = r['shadows']*ss(B, a, b)*(1-ss(B, c, d))
    e, f = p['hl_band']
    g = g-r['highlights']*ss(B, e, f)
    if r['face_fix']:
        _, each = face_mask(Y.shape, [x['box'] for x in r['face_fix']], full_shape)
        for m, fx in zip(each, r['face_fix']):
            if m.max() < 1e-3: continue
            inner = B[m > 0.5]
            top = float(np.percentile(inner, 72)) if inner.size else float(np.median(B))
            dark = ss(top-B, 0.03, 0.26)     # 0 на освітленій частині лиця, 1 у тіні
            g = g+m*(fx['lift']*(p['face_lift_flat']+(1-p['face_lift_flat'])*dark)
                     + fx['shadow']*dark)
    return g

def zone_gain(zones, shape, full_shape):
    hs, ws = shape; sy, sx = hs/full_shape[0], ws/full_shape[1]
    yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float32)
    ev = np.zeros(shape, np.float32); masks = []
    for z in zones or []:
        x, y, w, h = z['x']*ws, z['y']*hs, z['w']*ws, z['h']*hs
        fe = float(z.get('feather', 0.3))
        if z.get('shape', 'rect') == 'ellipse':
            dd = np.sqrt(((xx-(x+w/2))/max(w/2, 1))**2+((yy-(y+h/2))/max(h/2, 1))**2)
            m = 1-ss(dd, max(1-fe, 0.05), 1.0)
        else:
            fx, fy = max(w*fe, 1), max(h*fe, 1)
            m = (ss(xx, x-fx, x+fx)*(1-ss(xx, x+w-fx, x+w+fx))
                 * ss(yy, y-fy, y+fy)*(1-ss(yy, y+h-fy, y+h+fy)))
        ev += m*float(z.get('ev', 0.0)); masks.append((m, z))
    return ev, masks

def shoulder(lin, knee):
    """М'яке стиснення світел. Рахується лише там, де воно потрібне."""
    mx = chmax(lin); m = mx > knee
    if not m.any(): return lin
    v = mx[m]
    out = lin.copy()
    out[m] *= ((knee+(1-knee)*(1-np.exp(-(v-knee)/(1-knee))))/v)[:, None]
    return out

# ------------------------------------------------------------------ аналіз
def _levels(ys, r, p):
    """Рівні з гістограми: чорна точка, біла точка, середні тони."""
    q = np.percentile(ys, [0.5, 50, 99.5])
    r['black'] = round(float(np.clip(q[0]-p['black_target'], 0, p['black_cap'])), 3)
    r['white'] = round(float(np.clip(1+(p['white_target']/max(q[2], 0.35)-1)*p['white_pull'],
                                     1.0, p['white_cap'])), 3)
    m = float(np.clip(q[1]*r['white'], 0.04, 0.96))
    tgt = float(np.clip(m+(p['mid_target']-m)*p['mid_pull'], 0.04, 0.985))
    r['gamma'] = round(float(np.clip(np.log(tgt)/np.log(m), p['gamma_lo'], p['gamma_hi'])), 3)


def analyze(rgb8, ref=REF_SKIN, p=P, series_cast=None, cache=None):
    if cache is not None:
        sm = cv2.resize(rgb8, cache['sm_size'], interpolation=cv2.INTER_AREA)
        faces, chroma, nf = cache['faces'], cache['chroma'], cache['nf']
    else:
        sm, faces = detect_faces(rgb8)
        chroma, nf = skin_chroma(sm, faces)
    r = dict(faces=len(faces), skin_faces=nf, face_fix=[],
             face_box=[f['box'] for f in faces])

    # --- 1. колір: оцінка по цьому фото тим вагоміша, чим більше на ньому облич ---
    pc = cast_of(chroma, ref) if chroma is not None else None
    if series_cast is not None and pc is not None:
        w = p['photo_weight']*nf/(nf+p['conf_n'])
        t = series_cast[0]+w*(pc[0]-series_cast[0]); n = series_cast[1]+w*(pc[1]-series_cast[1])
        r['cast_src'] = 'серія+фото'
    elif series_cast is not None:
        t, n = series_cast; r['cast_src'] = 'серія'
    elif pc is not None:
        w = nf/(nf+p['conf_n']); t, n = pc[0]*w, pc[1]*w; r['cast_src'] = 'шкіра'
    else:
        t, n = 0.0, 0.0; r['cast_src'] = 'немає облич'
    r['temp'] = round(-soft(t, p['cast_dead'], p['cast_full'], p['cast_strength'], p['cast_cap']), 3)
    r['tint'] = round(+soft(n, p['cast_dead'], p['cast_full'], p['cast_strength'], p['cast_cap']), 3)
    lin0 = s2l8(sm)*corr_gains(r['temp'], r['tint'])

    # --- 2. стартова експозиція за світлотою шкіри (або за сценою) ---
    idx = skin_index(sm, faces)
    if idx is not None:
        L0 = lightness(np.median(lin0[idx[0], idx[1]], 0)); r['skinL'] = round(L0, 1)
        dv = p['face_L']-L0
        dead = p['dead_lo'] if dv > 0 else p['dead_hi']
        ev = float(np.clip((dv-np.sign(dv)*dead)/p['ev_div'], -p['ev_cap'], p['ev_cap'])) if abs(dv) > dead else 0.0
        r['ev_src'] = 'шкіра'
    else:
        med = float(l2s(np.median(np.maximum(luma(lin0), 1e-5))))
        ev = float(np.clip((p['scene_med']-med)*1.8, -p['scene_ev_cap'], p['scene_ev_cap'])) \
            if abs(med-p['scene_med']) > p['scene_dead'] else 0.0
        r['ev_src'] = 'сцена'
    r['ev'] = round(ev, 2)

    # --- 3. скільки піднімати напівтінь і тиснути світла ---
    Y = np.maximum(luma(lin0*(2.0**ev)), 1e-5)
    bs = base_srgb(Y, p)
    dark = float(np.mean(np.clip((p['sh_hi']-bs)/(p['sh_hi']-p['sh_lo']), 0, 1)*(bs > p['sh_lo'])))
    brig = float(np.mean(np.maximum(bs-p['hl_ref'], 0))/(1-p['hl_ref']))
    r['shadows'] = round(float(np.clip(dark*p['sh_k'], 0, p['sh_cap'])), 2)
    r['highlights'] = round(float(np.clip(brig*p['hl_k'], 0, p['hl_cap'])), 2)
    r['contrast'] = 0.0; r['sat'] = 0.0
    _levels(l2s(np.clip(luma(shoulder(lin0*(2.0**ev)*(2.0**local_gain(Y, r, p, rgb8.shape))[..., None],
                                      p['knee'])), 0, 1)), r, p)

    # --- 4. обличчя: підсвічуємо тіньову частину, освітлену майже не чіпаємо ---
    def pre(ev):
        lin = lin0*(2.0**ev); Yl = np.maximum(luma(lin), 1e-5)
        l2 = shoulder(lin*(2.0**local_gain(Yl, r, p, rgb8.shape))[..., None], p['knee'])
        return l2s(np.clip(luma(l2), 0, 1))
    ys = pre(ev)
    fm, each = face_mask(ys.shape, [f['box'] for f in faces], rgb8.shape)
    fix = []
    for f, m in zip(faces, each):
        px = ys[m > 0.45]
        if px.size < 40: continue
        lo, md, hi = (float(v) for v in np.percentile(px, [25, 50, 80]))
        lift = float(np.clip((p['face_target']-hi)*p['face_lift_k'], 0, p['face_lift_cap']))
        shd = float(np.clip((md-lo)*p['face_shadow_k'], 0, p['face_shadow_cap']))
        if lift > 0.02 or shd > 0.05:
            fix.append(dict(box=f['box'], lift=round(lift, 2), shadow=round(shd, 2)))
    r['face_fix'] = fix

    # --- 5. калібрування: середніми тонами доводимо світлоту шкіри до цілі
    #        вже на готовому результаті. Експозиція лишається тим, чим була. ---
    k = min(1.0, p['cal_px']/max(sm.shape[:2]))
    th, tw = max(int(sm.shape[0]*k), 60), max(int(sm.shape[1]*k), 60)
    tiny = cv2.resize(sm, (tw, th), interpolation=cv2.INTER_AREA)
    tidx = skin_index(tiny, faces, scale=tw/sm.shape[1])
    out = None
    for _ in range(p['cal_iter']):
        _levels(pre(r['ev']), r, p)
        out = render(tiny, r, p)
        if tidx is None: break
        L = skin_L(out, tidx)
        # не точка, а діапазон: піднімаємо лише темні обличчя, опускаємо лише вибиті
        d = p['cal_lo']-L if L < p['cal_lo'] else (p['cal_hi']-L if L > p['cal_hi'] else 0.0)
        if abs(d) < p['cal_tol']: break
        r['ev'] = round(float(np.clip(r['ev']+d/18.0, -1.0, 1.2)), 3)
    r['skinL_out'] = round(skin_L(out, tidx), 1) if tidx is not None else None

    # --- 6. стеля яскравості лиця: освітлена частина не повинна вибілюватись ---
    if fix and tidx is not None:
        yo = l2s(np.clip(luma(s2l8(out)), 0, 1))
        fmo, eo = face_mask(yo.shape, [f['box'] for f in fix], rgb8.shape)
        for k, (f, m) in enumerate(zip(fix, eo)):
            px = yo[m > 0.45]
            if px.size < 40: continue
            over = float(np.percentile(px, 92))-p['face_ceil']
            if over > 0:
                f['lift'] = round(max(f['lift']-over*3.0, 0.0), 2)
                f['shadow'] = round(max(f['shadow']-over*1.5, 0.0), 2)
        out = render(tiny, r, p)

    # --- 7. контраст і насиченість міряємо на готовому результаті ---
    yo = l2s(np.clip(luma(s2l8(out)), 0, 1))
    q5, q95 = (float(v) for v in np.percentile(yo, [5, 95]))
    r['contrast'] = round(float(np.clip((p['spread_target']-(q95-q5))/0.33, 0, p['contrast_cap'])), 2)
    o = out.astype(np.float32)/255.0; mx = chmax(o); mn = chmin(o)
    sel = (mx > 0.12) & (mx < 0.97)
    sat_now = float(np.median(((mx-mn)/np.maximum(mx, 1e-5))[sel])) if sel.any() else p['sat_target']
    r['sat_now'] = round(sat_now, 3)
    r['sat'] = round(float(np.clip(p['sat_base']+(p['sat_target']-sat_now)*p['sat_k'], 0, p['sat_cap'])), 2)

    # --- 8. захист світел: пересвітів має бути не більше, ніж в оригіналі плюс запас ---
    lvl = p['clip_level']
    was = float((chmax(tiny) >= lvl).mean())
    for _ in range(4):
        now = float((chmax(render(tiny, r, p)) >= lvl).mean())
        r['clip'] = round(now*100, 2)
        if now <= was+p['clip_budget']: break
        r['highlights'] = round(min(r['highlights']+0.22, p['hl_cap']+0.5), 2)
    r['clip_was'] = round(was*100, 2)
    return r


# ------------------------------------------------------------------ відтворення
def render(rgb8, r, p=P, zones=None):
    H, W = rgb8.shape[:2]
    lin = s2l8(rgb8)*corr_gains(r['temp'], r['tint'])*(2.0**r['ev'])
    sw, sh = max(int(W*p['map_scale']), 48), max(int(H*p['map_scale']), 48)
    Ysm = cv2.resize(luma(lin), (sw, sh), interpolation=cv2.INTER_AREA)
    gv = local_gain(np.maximum(Ysm, 1e-5), r, p, (H, W))
    zev, zmasks = zone_gain(zones, (sh, sw), (H, W))
    lin = shoulder(lin*cv2.resize(2.0**(gv+zev), (W, H), interpolation=cv2.INTER_LINEAR)[..., None], p['knee'])

    srgb = l2s(lin); Y = np.maximum(luma(srgb), 1e-5)
    t = np.clip((Y-r['black'])/(1-r['black']), 0, 1) if r['black'] > 0.002 else np.clip(Y, 0, 1)
    w = r.get('white', 1.0)
    if w > 1.001:                      # біла точка: світла підтягуються, без обрізання
        t = t*w; t = np.where(t <= 0.82, t, 0.82+(1-0.82)*(1-np.exp(-(t-0.82)/(1-0.82))))
    g = r.get('gamma', 1.0)
    if abs(g-1) > 0.002: t = np.power(np.clip(t, 0, 1), g)      # середні тони
    if r['contrast'] > 0.01:
        c = r['contrast']
        t = (1-c)*t+c*(0.5-np.sin(np.arcsin(np.clip(1-2*t, -1, 1))/3))
    srgb = srgb*(t/Y)[..., None]            # верх поки не обрізаємо: плече нижче

    # тепле підсвічування: жовто-оранжеві тони (шкіра, дерево, листя) стають світлішими
    if p['skin_lum'] > 0.005:
        # маска рахується у повному розмірі: зменшена розмивається через краї
        # і піднімає сусідні світлі ділянки, через що потім доводиться
        # сильніше тиснути світла і лице стає пласким
        mx = chmax(srgb); mn = chmin(srgb); d = np.maximum(mx-mn, 1e-5)
        isr = (srgb[..., 0] >= mx-1e-6)
        hue = 60.0*(srgb[..., 1]-srgb[..., 2])/d
        a1, a2, a3, a4 = p['skin_hue']
        m = (isr*ss(hue, a1, a2)*(1-ss(hue, a3, a4))*ss(d/np.maximum(mx, 1e-5), 0.06, 0.16)
             * ss(mx, 0.10, 0.22)*(1-ss(mx, 0.85, 0.99)))
        srgb = srgb*(1+p['skin_lum']*m)[..., None]

    # насиченість: слабконасичене піднімається більше, шкіра отримує невелику добавку
    if r['sat'] > 0.01 or r.get('face_box') or any(z.get('sat') for _, z in zmasks):
        mx = chmax(srgb); mn = chmin(srgb)
        sat = (mx-mn)/np.maximum(mx, 1e-5)
        f = 1+r['sat']*(1-ss(sat, *p['sat_band']))**1.2
        f = 1+(f-1)*(1-ss(mx, 0.85, 0.99))      # майже білі ділянки не насичуємо
        if r.get('face_box'):
            fm, _ = face_mask((sh, sw), r['face_box'], (H, W), grow=(1.0, 1.2))
            fmf = cv2.resize(fm, (W, H), interpolation=cv2.INTER_LINEAR)
            f = f+fmf*p['skin_sat']*(1-ss(mx, 0.85, 0.99))   # щоб лице не було блідим
        for m, z in zmasks:
            if z.get('sat'):
                f = f+cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)*float(z['sat'])
        yl = luma(srgb)[..., None]
        srgb = yl+(srgb-yl)*f[..., None]

    # плече по каналах: усе, що вилізло за межу, плавно повертається в діапазон,
    # тому на яскравих костюмах лишається фактура замість білої плями
    k = p['top_knee']; mx = chmax(srgb); m = mx > k
    if m.any():
        v = mx[m]
        srgb[m] *= ((k+(1-k)*np.tanh((v-k)/(1-k)))/v)[:, None]
    return (np.clip(srgb, 0, 1)*255+0.5).astype(np.uint8)

def with_adjust(r, adj):
    """Ручні поправки поверх автоматичних значень. «mid» піднімає середні тони
    (це зворотна величина до гамми, тому віднімається)."""
    if not adj: return r
    lim = dict(ev=(-2, 2), temp=(-.5, .5), tint=(-.5, .5), shadows=(0, 2), highlights=(0, 2),
               contrast=(-.5, .9), sat=(-.8, .8), black=(0, .12))
    r = dict(r)
    for k, (lo, hi) in lim.items():
        if k in adj: r[k] = float(np.clip(float(r.get(k, 0))+float(adj[k]), lo, hi))
    if 'mid' in adj:
        r['gamma'] = float(np.clip(float(r.get('gamma', 1.0))-float(adj['mid']), 0.55, 1.5))
    if 'white' in adj:
        r['white'] = float(np.clip(float(r.get('white', 1.0))+float(adj['white']), 1.0, 1.35))
    return r


def process(path_in, path_out, ref=REF_SKIN, p=P, zones=None, series_cast=None):
    from PIL import Image
    im = Image.open(path_in); icc = im.info.get('icc_profile')
    a = np.asarray(im.convert('RGB'))
    r = analyze(a, ref, p, series_cast)
    out = render(a, r, p, zones)
    if path_out: Image.fromarray(out).save(path_out, quality=95, subsampling=0, icc_profile=icc)
    return r
