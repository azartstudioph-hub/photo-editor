"""Локальний застосунок: перегляд «до/після», ручні поправки, зони, поворот,
обробка всієї папки і окреме збереження результату.
Запуск:  python app.py [шлях_до_папки_з_фото]
Оригінали ніколи не змінюються."""
import os, sys, json, io, shutil, threading, webbrowser, traceback
from concurrent.futures import ProcessPoolExecutor
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enhance as E, runner as R, imageio as IO

HERE = E.res_dir()
SETTINGS_FILE = 'azart-settings.json'
TMP_DIR = '__azart_tmp'
PREVIEW_MAX = 1280
LOCK = threading.Lock()
STATE = dict(src=None, files=[], small={}, cast={}, analysis={}, settings={}, job=None, workers=None)


# ------------------------------------------------------------------ налаштування
def load_settings(src):
    p = os.path.join(src, SETTINGS_FILE)
    if os.path.exists(p):
        try: return json.load(open(p, encoding='utf-8'))
        except Exception: pass
    return {}


def save_settings():
    if not STATE['src']: return
    data = {os.path.basename(f): s for f, s in STATE['settings'].items()
            if s.get('zones') or s.get('adjust') or s.get('rotate')}
    p = os.path.join(STATE['src'], SETTINGS_FILE)
    if data: json.dump(data, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    elif os.path.exists(p): os.remove(p)


def st_of(f):
    return STATE['settings'].setdefault(f, dict(zones=[], adjust={}, rotate=0))


def open_folder(src):
    files = R.files_in(src)
    STATE.update(src=src, files=files, small={}, cast={}, analysis={}, settings={}, job=None)
    saved = load_settings(src)
    for f in files:
        s = saved.get(os.path.basename(f)) or {}
        STATE['settings'][f] = dict(zones=s.get('zones', []), adjust=s.get('adjust', {}),
                                    rotate=int(s.get('rotate', 0)))
    threading.Thread(target=prepare_casts, daemon=True).start()
    return files


def prepare_casts():
    """Колір зйомок рахується один раз по вибірці кадрів — не по всій папці."""
    try:
        with LOCK:
            if STATE['cast'] or not STATE['files']: return
            files = list(STATE['files'])
        c = R.casts_for(files)
        STATE['cast'] = c
        STATE['analysis'].clear()
    except Exception:
        traceback.print_exc()


# ------------------------------------------------------------------ прев'ю
def small_image(f):
    key = (f, st_of(f)['rotate'])
    if STATE['small'].get('key') != key:
        arr, _ = IO.load(f, st_of(f)['rotate'], max_side=PREVIEW_MAX)
        STATE['small'] = dict(key=key, arr=arr)
    return STATE['small']['arr']


def analysis(f):
    key = (f, st_of(f)['rotate'], bool(STATE['cast']))
    if STATE['analysis'].get(f, (None,))[0] != key:
        a = small_image(f)
        STATE['analysis'][f] = (key, E.analyze(a, series_cast=STATE['cast'].get(f)))
    return STATE['analysis'][f][1]


def jpeg(arr, q=86):
    b = io.BytesIO(); Image.fromarray(arr).save(b, 'JPEG', quality=q); return b.getvalue()


# ------------------------------------------------------------------ пакетна обробка
def batch(workers=None):
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[v] = '1'        # процеси-помічники не множать потоки
    src = STATE['src']; tmp = os.path.join(src, TMP_DIR)
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp, exist_ok=True)
    files = list(STATE['files'])
    STATE['job'] = dict(stage='process', done=False, i=0, n=len(files), name='', errors=[])
    try:
        if not STATE['cast']: prepare_casts()
        args = [(f, os.path.join(tmp, os.path.basename(f)), STATE['cast'].get(f),
                 st_of(f)['zones'], st_of(f)['adjust'], st_of(f)['rotate']) for f in files]
        n = int(workers or STATE.get('workers') or R.default_workers())
        n = max(1, min(n, 8))
        STATE['job']['workers'] = n
        if n > 1:
            with ProcessPoolExecutor(max_workers=n, initializer=R.init_worker) as ex:
                for k, err in enumerate(ex.map(R.work, args, chunksize=1)):
                    STATE['job'].update(i=k+1, name=os.path.basename(files[k]))
                    if err: STATE['job']['errors'].append(err)
        else:
            R.init_worker()
            for k, a in enumerate(args):
                err = R.work(a)
                STATE['job'].update(i=k+1, name=os.path.basename(files[k]))
                if err: STATE['job']['errors'].append(err)
        STATE['job'].update(done=True, tmp=tmp)
    except Exception:
        STATE['job'] = dict(stage='process', done=True, error=traceback.format_exc())


def save_results(folder):
    src = STATE['src']; tmp = os.path.join(src, TMP_DIR)
    dst = folder if os.path.isabs(folder) else os.path.join(src, folder)
    STATE['job'] = dict(stage='save', done=False, i=0, n=0, name='')
    try:
        names = sorted(os.listdir(tmp)) if os.path.isdir(tmp) else []
        os.makedirs(dst, exist_ok=True)
        STATE['job']['n'] = len(names)
        for k, nm in enumerate(names):
            shutil.move(os.path.join(tmp, nm), os.path.join(dst, nm))
            STATE['job'].update(i=k+1, name=nm)
        shutil.rmtree(tmp, ignore_errors=True)
        STATE['job'].update(done=True, dst=dst)
    except Exception:
        STATE['job'] = dict(stage='save', done=True, error=traceback.format_exc())


# ------------------------------------------------------------------ сервер
class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a): pass

    def _send(self, code, body, ctype='application/json; charset=utf-8', extra=None):
        self.send_response(code); self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode('utf-8'))

    def _pack(self):
        return dict(src=STATE['src'], files=[os.path.basename(f) for f in STATE['files']],
                    settings={os.path.basename(f): STATE['settings'][f] for f in STATE['files']},
                    ready=bool(STATE['cast']), cores=os.cpu_count() or 2,
                    workers=STATE.get('workers') or R.default_workers())

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        try:
            if u.path in ('/', '/index.html'):
                return self._send(200, open(os.path.join(HERE, 'ui.html'), 'rb').read(),
                                  'text/html; charset=utf-8')
            if u.path == '/api/state': return self._json(self._pack())
            if u.path == '/api/open':
                p = q.get('path', [''])[0].strip().strip('"')
                if not os.path.isdir(p): return self._json(dict(error='Папку не знайдено: '+p), 400)
                open_folder(p); return self._json(self._pack())
            if u.path in ('/api/before', '/api/after'):
                f = os.path.join(STATE['src'], q['file'][0])
                if f not in STATE['settings']: return self._json(dict(error='немає файлу'), 404)
                with LOCK:
                    if u.path == '/api/before':
                        return self._send(200, jpeg(small_image(f)), 'image/jpeg')
                    s = st_of(f)
                    r = E.with_adjust(analysis(f), s.get('adjust'))
                    body = jpeg(E.render(small_image(f), r, zones=s.get('zones')))
                info = {k: r.get(k) for k in ('temp', 'tint', 'ev', 'shadows', 'highlights',
                                              'gamma', 'white', 'contrast', 'sat', 'black')}
                return self._send(200, body, 'image/jpeg',
                                  {'X-Info': json.dumps(info), 'X-Token': q.get('tok', [''])[0]})
            if u.path == '/api/progress':
                return self._json(STATE['job'] or dict(done=True, stage='none'))
        except Exception:
            return self._json(dict(error=traceback.format_exc()), 500)
        self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        u = urlparse(self.path)
        data = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        try:
            if u.path == '/api/settings':
                f = os.path.join(STATE['src'], data['file'])
                s = st_of(f)
                for k in ('zones', 'adjust', 'rotate'):
                    if k in data: s[k] = data[k]
                save_settings(); return self._json(dict(ok=True))
            if u.path == '/api/apply_all':
                s = st_of(os.path.join(STATE['src'], data['from']))
                for f in STATE['files']:
                    st_of(f)['adjust'] = dict(s.get('adjust', {}))
                save_settings(); return self._json(self._pack())
            if u.path == '/api/process':
                if data.get('workers'): STATE['workers'] = int(data['workers'])
                threading.Thread(target=batch, daemon=True).start()
                return self._json(dict(ok=True))
            if u.path == '/api/save':
                threading.Thread(target=save_results, args=(data.get('folder') or 'оброблені',),
                                 daemon=True).start()
                return self._json(dict(ok=True))
        except Exception:
            return self._json(dict(error=traceback.format_exc()), 500)
        self._send(404, b'not found', 'text/plain')


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    port = int(os.environ.get('AZART_PORT', '8765'))
    R.lower_priority()
    if os.path.isdir(src): open_folder(src)
    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    url = 'http://127.0.0.1:%d/' % port
    print('AZART — обробка фото. Відкрийте в браузері:', url)
    if not os.environ.get('AZART_NOBROWSER'):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: pass


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
